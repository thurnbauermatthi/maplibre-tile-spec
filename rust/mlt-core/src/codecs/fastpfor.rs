use fastpfor::{AnyLenCodec as _, FastPFor256};
use usize_cast::IntoUsize as _;

use crate::decoder::FastPForKind;
use crate::{Decoder, MltError, MltResult};

/// Decode `FastPFOR`-compressed data using the composite codec protocol.
///
/// The Java MLT encoder uses `Composition(FastPFOR(), VariableByte())`, matching
/// the C++ `CompositeCodec<FastPFor<8>, VariableByte>`. The wire format is:
///
/// 1. First u32 = number of compressed u32 words from the primary codec (`FastPFor`)
/// 2. Next N u32 words = primary codec (`FastPFor`) compressed data
/// 3. Remaining u32 words = secondary codec (`VByte`) compressed data
///
/// [`FastPForKind`] fixes both the block size and the order the words are stored in.
pub fn decode_fastpfor(
    data: &[u8],
    num_values: u32,
    kind: FastPForKind,
    dec: &mut Decoder,
) -> MltResult<Vec<u32>> {
    if !cfg!(feature = "fastpfor") {
        return Err(MltError::NotImplemented(
            "FastPFOR decoding, which this build leaves out",
        ));
    }
    if num_values == 0 {
        // FIXME: eventually there should not be a header anywhere at all
        return if data.is_empty() {
            Ok(vec![])
        } else {
            Err(MltError::InvalidFastPforByteLength(0))
        };
    }

    let (words, rest) = data.as_chunks::<4>();
    if !rest.is_empty() {
        return Err(MltError::InvalidFastPforByteLength(data.len()));
    }
    dec.consume_items::<u32>(words.len())?;

    let mut result = Vec::new();
    match kind {
        FastPForKind::Block256Be => {
            let input: Vec<u32> = words.iter().copied().map(u32::from_be_bytes).collect();
            FastPFor256::default().decode(&input, &mut result, Some(num_values))?;
        }
    }

    let adjustment = result
        .len()
        .checked_sub(num_values.into_usize())
        .and_then(|v| u32::try_from(v).ok())
        .expect("infallible: the codec rejects a short decode, so the excess is block padding");

    dec.adjust(adjustment);
    result.truncate(num_values.into_usize());

    Ok(result)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::test_helpers::{dec, starved_dec};

    const KIND: FastPForKind = FastPForKind::Block256Be;

    fn encode(data: &[u32]) -> Vec<u8> {
        let mut words = Vec::new();
        FastPFor256::default().encode(data, &mut words).unwrap();
        words.iter().flat_map(|w| w.to_be_bytes()).collect()
    }

    proptest! {
        #[test]
        fn test_fastpfor_roundtrip(data: Vec<u32>) {
            // FastPFor produces a non-empty output (VByte header) even for empty input,
            // but decode_fastpfor requires zero bytes when num_values == 0 - consistent
            // with how PhysicalEncoder guards `if !values.is_empty()`.
            prop_assume!(!data.is_empty());
            let encoded = encode(&data);
            let decoded = decode_fastpfor(&encoded, data.len().try_into().unwrap(), KIND, &mut dec()).unwrap();
            prop_assert_eq!(data, decoded);
        }
    }

    #[test]
    fn test_decode_fastpfor_empty() {
        let decoded = decode_fastpfor(&[], 0, KIND, &mut dec()).unwrap();
        assert_eq!(decoded, [] as [u32; 0]);
    }

    #[test]
    fn a_zero_value_stream_carrying_a_payload_is_rejected() {
        let err = decode_fastpfor(&[0, 0, 0, 0], 0, KIND, &mut dec()).unwrap_err();
        assert!(
            matches!(err, MltError::InvalidFastPforByteLength(0)),
            "{err:?}"
        );
    }

    #[test]
    fn a_payload_that_is_not_whole_words_is_rejected() {
        let err = decode_fastpfor(&[0, 0, 0], 1, KIND, &mut dec()).unwrap_err();
        assert!(
            matches!(err, MltError::InvalidFastPforByteLength(3)),
            "{err:?}"
        );
    }

    #[test]
    fn a_declared_count_the_codec_cannot_meet_is_rejected() {
        let data: Vec<u32> = (0..500).map(|i| i * 7 + 3).collect();
        let err = decode_fastpfor(&encode(&data), 600, KIND, &mut dec()).unwrap_err();
        assert!(
            matches!(err, MltError::FastPfor(_)),
            "the codec must reject a short decode itself: {err:?}"
        );
    }

    #[test]
    fn decoding_past_the_memory_budget_is_rejected() {
        let data: Vec<u32> = (0..500).map(|i| i * 7 + 3).collect();
        let err = decode_fastpfor(&encode(&data), 500, KIND, &mut starved_dec()).unwrap_err();
        assert!(
            matches!(err, MltError::MemoryLimitExceeded { .. }),
            "{err:?}"
        );
    }
}
