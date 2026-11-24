//! The two ways a v2 layer says which features have a value.
//!
//! A bitmap costs `ceil(N/8)` bytes whatever it holds, which is what a general-purpose
//! compressor wants: runs of equal bytes cost it almost nothing. A tile read without one
//! can store the bitmap [`ByteRle`](PresenceCoding::ByteRle)-coded instead, as v1 does.
//!
//! Both are self-delimiting given `N`, so nothing in the wire format writes a length for them.

use std::borrow::Cow;

use bitvec::order::Lsb0;
use bitvec::slice::BitSlice;
use bitvec::vec::BitVec;
use bitvec::view::BitView as _;
use usize_cast::IntoUsize as _;

use crate::MltError::{self, PresenceRunOverflow};
use crate::codecs::rle::encode_byte_rle;
use crate::decoder::{BoolLogical, RleMeta};
use crate::utils::take;
use crate::{MltRefResult, MltResult};

/// Charges the bits a coding builds, so a tile cannot ask for an allocation far larger
/// than the bytes it spent asking.
pub(crate) trait PresenceBudget {
    /// Reserve room for `count` bits, or fail rather than allocate it.
    fn reserve_bits(&mut self, count: u32) -> MltResult<()>;
}

/// For the readers that hold no budget of their own - the dump walker and the renderer,
/// which are handed a payload whose length already bounds what it can name.
pub(crate) struct Unmetered;

impl PresenceBudget for Unmetered {
    fn reserve_bits(&mut self, _count: u32) -> MltResult<()> {
        Ok(())
    }
}

/// How a bitfield is stored, numbered as the `Bool` stream family numbers its logical field.
/// A presence nibble names one as `code + 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
#[repr(u8)]
pub enum PresenceCoding {
    /// `ceil(N/8)` bytes, LSB-first. Bits past `N` are padding and are ignored.
    Bitmap = 0,
    /// The `Bitmap` bytes, ORC byte-RLE coded as v1 codes its present streams.
    /// An extension: a tile served with transport compression stores the bitmap as it is.
    ByteRle = 1,
}

impl PresenceCoding {
    /// Every coding, in the order a tie between them is broken.
    pub(crate) const ALL: [Self; 2] = [Self::Bitmap, Self::ByteRle];

    /// The coding a `Bool` stream's logical field numbers `code`, or [`None`] for one this version has no meaning for.
    #[must_use]
    pub(crate) fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Bitmap),
            1 => Some(Self::ByteRle),
            _ => None,
        }
    }

    /// The shared logical encoding of `count` bits stored in this coding.
    #[must_use]
    pub(crate) fn logical(self, count: u32) -> BoolLogical {
        match self {
            Self::Bitmap => BoolLogical::None,
            Self::ByteRle => BoolLogical::ByteRle(RleMeta::Interleaved {
                num_rle_values: count,
            }),
        }
    }

    /// The coding a bool stream's logical encoding names.
    #[must_use]
    pub(crate) fn of_logical(logical: BoolLogical) -> Self {
        match logical {
            BoolLogical::None => Self::Bitmap,
            BoolLogical::ByteRle(_) => Self::ByteRle,
        }
    }
}

/// `bits` packed LSB-first into `ceil(len/8)` bytes.
fn bitmap(bits: &[bool]) -> Vec<u8> {
    let mut out = vec![0; bits.len().div_ceil(8)];
    for (i, &bit) in bits.iter().enumerate() {
        if bit {
            out[i / 8] |= 1 << (i % 8);
        }
    }
    out
}

/// Write `bits` in `coding`, appending to `out`.
pub(crate) fn write(out: &mut Vec<u8>, bits: &[bool], coding: PresenceCoding) {
    match coding {
        PresenceCoding::Bitmap => out.extend_from_slice(&bitmap(bits)),
        PresenceCoding::ByteRle => {
            let mut coded = Vec::new();
            out.extend_from_slice(encode_byte_rle(&bitmap(bits), &mut coded));
        }
    }
}

/// What `bits` would take in `coding`, without keeping it.
#[must_use]
pub(crate) fn size(bits: &[bool], coding: PresenceCoding) -> usize {
    match coding {
        PresenceCoding::Bitmap => bits.len().div_ceil(8),
        PresenceCoding::ByteRle => encode_byte_rle(&bitmap(bits), &mut Vec::new()).len(),
    }
}

/// The smallest coding among `allowed` for `bits`, ties broken toward the lowest code so that
/// the same mask always leaves the same bytes.
#[must_use]
pub(crate) fn smallest(bits: &[bool], allowed: &[PresenceCoding]) -> PresenceCoding {
    allowed
        .iter()
        .copied()
        .min_by_key(|&c| (size(bits, c), c as u8))
        .unwrap_or(PresenceCoding::Bitmap)
}

/// One run of a byte-RLE payload.
enum ByteRun<'a> {
    Literal(&'a [u8]),
    Repeat(u8, usize),
}

/// Walk the byte-RLE runs that make up `num_bytes` bitmap bytes, calling `f` with each run and
/// the bitmap offset it starts at, and return how many input bytes they took.
/// A run that would end past `num_bytes` is an error, so the payload delimits itself exactly.
fn walk_byte_rle<'a>(
    input: &'a [u8],
    num_bytes: usize,
    count: u32,
    mut f: impl FnMut(usize, ByteRun<'a>),
) -> MltResult<usize> {
    let (mut pos, mut produced) = (0usize, 0usize);
    while produced < num_bytes {
        let control = *input.get(pos).ok_or(MltError::UnableToTake(count))?;
        pos += 1;
        let (run, len) = if control >= 128 {
            let len = usize::from(control ^ 0xFF) + 1;
            let bytes = input
                .get(pos..pos + len)
                .ok_or(MltError::UnableToTake(count))?;
            pos += len;
            (ByteRun::Literal(bytes), len)
        } else {
            let len = usize::from(control) + 3;
            let &value = input.get(pos).ok_or(MltError::UnableToTake(count))?;
            pos += 1;
            (ByteRun::Repeat(value, len), len)
        };
        if produced + len > num_bytes {
            return Err(PresenceRunOverflow(count));
        }
        f(produced, run);
        produced += len;
    }
    Ok(pos)
}

/// Read `count` presence bits stored in `coding`, charging what it builds to `budget`.
///
/// A bitmap is borrowed from the tile bytes, and needs `ceil(count/8)` of them to be there
/// at all, so the input bounds it. A byte-RLE run of two bytes can ask for any `count` the
/// layer declared, so the allocation is reserved before it is made rather than trusted.
pub(crate) fn read<'a>(
    input: &'a [u8],
    count: u32,
    coding: PresenceCoding,
    budget: &mut dyn PresenceBudget,
) -> MltRefResult<'a, Cow<'a, BitSlice<u8, Lsb0>>> {
    let n = count.into_usize();
    match coding {
        PresenceCoding::Bitmap => {
            let (input, bytes) = take(input, count.div_ceil(8))?;
            Ok((input, Cow::Borrowed(&bytes.view_bits::<Lsb0>()[..n])))
        }
        PresenceCoding::ByteRle => {
            budget.reserve_bits(count)?;
            let num_bytes = n.div_ceil(8);
            let mut plain = vec![0u8; num_bytes];
            let used = walk_byte_rle(input, num_bytes, count, |at, run| match run {
                ByteRun::Literal(bytes) => plain[at..at + bytes.len()].copy_from_slice(bytes),
                ByteRun::Repeat(value, len) => plain[at..at + len].fill(value),
            })?;
            let mut bits = BitVec::<u8, Lsb0>::from_vec(plain);
            bits.truncate(n);
            Ok((&input[used..], Cow::Owned(bits)))
        }
    }
}

/// Split off the bytes `count` bits take in `coding`, without building them.
///
/// Every coding delimits itself, so this is how a reader finds where a stream's payload ends.
pub(crate) fn split(input: &[u8], count: u32, coding: PresenceCoding) -> MltRefResult<'_, &[u8]> {
    let len = match coding {
        PresenceCoding::Bitmap => count.div_ceil(8).into_usize(),
        PresenceCoding::ByteRle => {
            walk_byte_rle(input, count.into_usize().div_ceil(8), count, |_, _| {})?
        }
    };
    let (rest, payload) = take(input, u32::try_from(len)?)?;
    Ok((rest, payload))
}

/// How many of `count` bits `payload` marks present, without building them.
///
/// `payload` is exactly what [`split`] returned for `count` and `coding`.
pub(crate) fn popcount(payload: &[u8], count: u32, coding: PresenceCoding) -> MltResult<u32> {
    let n = count.into_usize();
    let num_bytes = n.div_ceil(8);
    // Bits past `n` in the last byte are padding.
    let valid = |at: usize, byte: u8| {
        let bits = (n - at * 8).min(8);
        (byte & (u8::MAX >> (8 - bits))).count_ones() as usize
    };
    let bits = match coding {
        PresenceCoding::Bitmap => payload
            .get(..num_bytes)
            .ok_or(MltError::UnableToTake(count))?
            .view_bits::<Lsb0>()[..n]
            .count_ones(),
        PresenceCoding::ByteRle => {
            let mut total = 0usize;
            walk_byte_rle(payload, num_bytes, count, |at, run| match run {
                ByteRun::Literal(bytes) => {
                    for (i, &b) in bytes.iter().enumerate() {
                        total += valid(at + i, b);
                    }
                }
                ByteRun::Repeat(value, len) => {
                    total += (value.count_ones() as usize) * (len - 1) + valid(at + len - 1, value);
                }
            })?;
            total
        }
    };
    Ok(u32::try_from(bits)?)
}

/// Read `bits` back out of whatever [`write`] put in, for a round-trip check.
#[cfg(test)]
pub(crate) fn round_trip(bits: &[bool], coding: PresenceCoding) -> Vec<bool> {
    let mut buf = Vec::new();
    write(&mut buf, bits, coding);
    assert_eq!(buf.len(), size(bits, coding), "size disagrees with write");
    let count = u32::try_from(bits.len()).unwrap();
    let (rest, out) = read(&buf, count, coding, &mut Unmetered).unwrap();
    assert!(rest.is_empty(), "{coding:?} left {} bytes", rest.len());
    out.iter().by_vals().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cases() -> Vec<Vec<bool>> {
        vec![
            vec![],
            vec![true],
            vec![false],
            vec![true; 9],
            vec![false; 9],
            // one contiguous present block, which byte-RLE is for
            (0..1000).map(|i| (200..600).contains(&i)).collect(),
            // a single present feature far in
            (0..1000).map(|i| i == 777).collect(),
            // alternating, the worst case for byte-RLE, where Bitmap wins
            (0..64).map(|i| i % 2 == 0).collect(),
            // ends on a present run inside the last byte
            (0..17).map(|i| i > 12).collect(),
            // a long mask with a few scattered bits
            (0..4096)
                .map(|i| matches!(i, 5 | 6 | 1000 | 3000))
                .collect(),
        ]
    }

    #[test]
    fn every_coding_round_trips() {
        for bits in cases() {
            for coding in PresenceCoding::ALL {
                assert_eq!(round_trip(&bits, coding), bits, "{coding:?} on {bits:?}");
            }
        }
    }

    #[test]
    fn smallest_is_the_smallest_allowed() {
        for bits in cases() {
            let picked = smallest(&bits, &PresenceCoding::ALL);
            let best = PresenceCoding::ALL
                .into_iter()
                .map(|c| size(&bits, c))
                .min()
                .unwrap();
            assert_eq!(size(&bits, picked), best, "on {bits:?}");
            assert_eq!(
                smallest(&bits, &[PresenceCoding::Bitmap]),
                PresenceCoding::Bitmap
            );
        }
    }

    #[test]
    fn a_tie_goes_to_the_lowest_code() {
        assert_eq!(
            smallest(&[true], &PresenceCoding::ALL),
            PresenceCoding::Bitmap
        );
    }

    #[test]
    fn byte_rle_beats_a_bitmap_on_one_block() {
        let bits: Vec<bool> = (0..4096).map(|i| (100..2000).contains(&i)).collect();
        assert_eq!(
            smallest(&bits, &PresenceCoding::ALL),
            PresenceCoding::ByteRle
        );
    }

    #[test]
    fn byte_rle_is_v1s_byte_rle_of_the_bitmap() {
        // 24 present values then 24 absent: three 0xFF bytes, then three 0x00 bytes.
        let bits: Vec<bool> = (0..48).map(|i| i < 24).collect();
        let mut buf = Vec::new();
        write(&mut buf, &bits, PresenceCoding::ByteRle);
        assert_eq!(buf, [0x00, 0xFF, 0x00, 0x00]);
    }

    #[test]
    fn a_run_past_the_bitmap_is_rejected() {
        // A repeat run of three bytes for a bitmap of two.
        assert!(matches!(
            read(&[0x00, 0xFF], 16, PresenceCoding::ByteRle, &mut Unmetered),
            Err(PresenceRunOverflow(16))
        ));
        assert!(split(&[0x00, 0xFF], 16, PresenceCoding::ByteRle).is_err());
    }

    #[test]
    fn only_codes_zero_and_one_name_a_coding() {
        let named: Vec<u8> = (0..=u8::MAX)
            .filter(|&c| PresenceCoding::from_code(c).is_some())
            .collect();
        assert_eq!(named, [0, 1]);
    }

    #[test]
    fn split_takes_exactly_what_write_made() {
        for bits in cases() {
            for coding in PresenceCoding::ALL {
                let mut buf = Vec::new();
                write(&mut buf, &bits, coding);
                buf.extend([0xAA, 0xBB]);
                let count = u32::try_from(bits.len()).unwrap();
                let (rest, payload) = split(&buf, count, coding).unwrap();
                assert_eq!(rest, [0xAA, 0xBB], "{coding:?} on {bits:?}");
                assert_eq!(payload.len(), size(&bits, coding), "{coding:?} on {bits:?}");
            }
        }
    }

    #[test]
    fn popcount_agrees_with_read() {
        for bits in cases() {
            for coding in PresenceCoding::ALL {
                let mut buf = Vec::new();
                write(&mut buf, &bits, coding);
                let count = u32::try_from(bits.len()).unwrap();
                let expected = u32::try_from(bits.iter().filter(|&&b| b).count()).unwrap();
                assert_eq!(
                    popcount(&buf, count, coding).unwrap(),
                    expected,
                    "{coding:?} on {bits:?}"
                );
            }
        }
    }

    #[test]
    fn popcount_ignores_padding_bits_past_the_count() {
        assert_eq!(
            popcount(&[0b1111_1111], 3, PresenceCoding::Bitmap).unwrap(),
            3
        );
        // A literal run of one byte, every bit set.
        assert_eq!(
            popcount(&[0xFF, 0b1111_1111], 3, PresenceCoding::ByteRle).unwrap(),
            3
        );
        // A repeat run of three 0xFF bytes over 20 bits.
        assert_eq!(
            popcount(&[0x00, 0xFF], 20, PresenceCoding::ByteRle).unwrap(),
            20
        );
    }

    #[test]
    fn split_rejects_a_truncated_payload() {
        assert!(split(&[1], 16, PresenceCoding::Bitmap).is_err());
        assert!(split(&[0xFE, 1], 16, PresenceCoding::ByteRle).is_err());
    }

    /// Counts the bits a reader was asked to build, standing in for a real budget.
    struct Counted(u32);
    impl PresenceBudget for Counted {
        fn reserve_bits(&mut self, count: u32) -> MltResult<()> {
            self.0 += count;
            Ok(())
        }
    }

    #[test]
    fn a_tiny_payload_cannot_ask_for_a_huge_allocation_unmetered() {
        // Two bytes naming four billion features: the budget has to hear about it
        // before anything is allocated, or a tile of nothing OOMs the reader.
        let mut budget = Counted(0);
        let _ = read(&[0x7F, 0], u32::MAX, PresenceCoding::ByteRle, &mut budget);
        assert_eq!(budget.0, u32::MAX);
    }

    struct Refused;
    impl PresenceBudget for Refused {
        fn reserve_bits(&mut self, count: u32) -> MltResult<()> {
            Err(MltError::MemoryLimitExceeded {
                limit: 0,
                used: 0,
                requested: count,
            })
        }
    }

    #[test]
    fn a_refused_budget_is_an_error() {
        assert!(matches!(
            read(&[0xFF, 0], 8, PresenceCoding::ByteRle, &mut Refused),
            Err(MltError::MemoryLimitExceeded { requested: 8, .. })
        ));
    }

    #[test]
    fn a_short_bitmap_is_rejected() {
        assert!(matches!(
            read(&[0], 9, PresenceCoding::Bitmap, &mut Unmetered),
            Err(MltError::UnableToTake(2))
        ));
    }
}
