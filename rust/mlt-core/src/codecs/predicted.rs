//! `Predicted` vertices: each vertex is guessed from the vertices decoded before it, or copied
//! from an earlier stretch of the same layer, and only what the guess misses is stored.
//!
//! The guesses follow the shape of map geometry and need nothing but the layer's topology:
//!
//! - a ring or line starts near where the previous ring or line started, or, for layers whose
//!   lines continue one another, where the previous run ended,
//! - the fourth vertex of a four-vertex ring completes a parallelogram,
//! - every other vertex is guessed as the vertex before it.
//!
//! Neighbouring polygons share borders and roads share junctions, so a stretch of vertices
//! that an earlier ring or line already holds is stored as a copy of it, forwards or backwards.
//!
//! The residuals go into one stream per kind of vertex and are stored a byte each, with larger
//! values escaped to a second plane, so the general-purpose compressor the tile is served with
//! sees homogeneous, byte-aligned symbols.
#![expect(
    clippy::many_single_char_names,
    reason = "v, s, i, j, k follow the notation of the specification"
)]

use usize_cast::IntoUsize as _;

use crate::codecs::varint::parse_varint;
use crate::decoder::{GeometryType, Levels, run_offsets};
use crate::{Decoder, MltError, MltResult};

/// Bit 0 of the flags byte: a copy-operations block follows.
const FLAG_COPIES: u8 = 1;
/// Bit 2 of the flags byte: a ring or line starts near the last vertex of the run before it,
/// instead of near its first vertex. Pays off where lines are stored as chained segments.
const FLAG_FROM_END: u8 = 4;
/// The shortest copy written; shorter stretches cost more as a copy than as residuals.
const MIN_COPY: usize = 2;
/// Earlier occurrences of a vertex the encoder tries as copy sources.
const COPY_CANDIDATES: usize = 8;
/// Values below this fit the low plane; the escape byte sends the rest to the high plane.
const ESCAPE: u64 = 0xFF;

/// What a run of the vertex sequence is, which decides how its vertices are guessed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunKind {
    Point,
    Line,
    Ring,
}

/// One run of the layer's vertex sequence: its length in vertices and its kind.
pub type Run = (u32, RunKind);

/// The residual streams, in the order the payload stores them.
const START: usize = 0;
const POINT: usize = 1;
const RING: usize = 2;
const CORNER: usize = 3;
const LINE1: usize = 4;
const LINE: usize = 5;
const CLASSES: usize = 6;

/// The vertex runs the topology spells out: one per ring, line part and point.
pub(crate) fn runs(
    types: &[GeometryType],
    levels: &Levels,
    vertex_count: usize,
) -> MltResult<Vec<Run>> {
    let Some(finest) = run_offsets(levels) else {
        // No topology: every vertex is a point of its own.
        if types.len() != vertex_count || types.iter().any(|t| !is_point(*t)) {
            return Err(MltError::MalformedPredicted("vertices without topology"));
        }
        return Ok(vec![(1, RunKind::Point); vertex_count]);
    };
    let coarser: Vec<&[u32]> = [&levels.geometries, &levels.parts, &levels.rings]
        .into_iter()
        .filter_map(|l| l.as_deref())
        .filter(|l| !std::ptr::eq(*l, finest))
        .collect();
    let run_count = finest.len().saturating_sub(1);
    let mut kinds = vec![None; run_count];
    for (feature, &t) in types.iter().enumerate() {
        let mut range = feature..feature + 1;
        for level in &coarser {
            let at = |i: usize| level.get(i).map(|&v| v.into_usize());
            range = match (at(range.start), at(range.end)) {
                (Some(s), Some(e)) if s <= e => s..e,
                _ => return Err(MltError::MalformedPredicted("topology out of range")),
            };
        }
        let kind = if is_point(t) {
            RunKind::Point
        } else if t.is_linestring() {
            RunKind::Line
        } else {
            RunKind::Ring
        };
        for run in range {
            *kinds
                .get_mut(run)
                .ok_or(MltError::MalformedPredicted("topology out of range"))? = Some(kind);
        }
    }
    let mut out = Vec::with_capacity(run_count);
    for (w, kind) in finest.windows(2).zip(kinds) {
        let len = w[1]
            .checked_sub(w[0])
            .ok_or(MltError::MalformedPredicted("decreasing run offsets"))?;
        out.push((
            len,
            kind.ok_or(MltError::MalformedPredicted("a run no feature owns"))?,
        ));
    }
    if finest.last().map_or(0, |&v| v.into_usize()) != vertex_count {
        return Err(MltError::MalformedPredicted(
            "the runs do not cover the vertices",
        ));
    }
    Ok(out)
}

fn is_point(t: GeometryType) -> bool {
    matches!(t, GeometryType::Point | GeometryType::MultiPoint)
}

type Xy = (i64, i64);

fn at(v: &[i32], i: usize) -> Xy {
    (i64::from(v[2 * i]), i64::from(v[2 * i + 1]))
}

fn sub(a: Xy, b: Xy) -> Xy {
    (a.0 - b.0, a.1 - b.1)
}

fn add(a: Xy, b: Xy) -> Xy {
    (a.0 + b.0, a.1 + b.1)
}

/// The class of vertex `i` of a run of `len` vertices and `kind`, past its first vertex.
fn class(kind: RunKind, len: usize, i: usize) -> usize {
    match kind {
        RunKind::Point => POINT,
        RunKind::Ring if len == 4 && i == 3 => CORNER,
        RunKind::Ring => RING,
        RunKind::Line if i == 1 => LINE1,
        RunKind::Line => LINE,
    }
}

/// The residual of vertex `i` of the run starting at vertex `s`, given every vertex before it.
/// `prev` is the vertex before the run, `start_guess` where the run is guessed to start.
fn residual(
    v: &[i32],
    s: usize,
    len: usize,
    kind: RunKind,
    i: usize,
    prev: Xy,
    start_guess: Xy,
) -> (usize, Xy) {
    let p = at(v, s + i);
    if i == 0 {
        return match kind {
            RunKind::Point => (POINT, sub(p, prev)),
            RunKind::Line | RunKind::Ring => (START, sub(p, start_guess)),
        };
    }
    let c = class(kind, len, i);
    let r = match c {
        CORNER => sub(p, sub(add(at(v, s), at(v, s + 2)), at(v, s + 1))),
        _ => sub(p, at(v, s + i - 1)),
    };
    (c, r)
}

fn zigzag(v: i64) -> u64 {
    #[expect(clippy::cast_sign_loss, reason = "zigzag")]
    let z = ((v << 1) ^ (v >> 63)) as u64;
    z
}

fn unzigzag(z: u64) -> i64 {
    #[expect(clippy::cast_possible_wrap, reason = "zigzag")]
    let v = (z >> 1) as i64 ^ -((z & 1) as i64);
    v
}

fn put_varint(mut v: u64, out: &mut Vec<u8>) {
    while v >= 0x80 {
        #[expect(clippy::cast_possible_truncation, reason = "low 7 bits")]
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    #[expect(clippy::cast_possible_truncation, reason = "below 0x80")]
    out.push(v as u8);
}

/// Per class, the low plane and the escaped high plane.
#[derive(Default)]
struct Planes {
    lo: Vec<u8>,
    hi: Vec<u8>,
}

impl Planes {
    fn push(&mut self, v: u64) {
        if v < ESCAPE {
            #[expect(clippy::cast_possible_truncation, reason = "below 0xFF")]
            self.lo.push(v as u8);
        } else {
            #[expect(clippy::cast_possible_truncation, reason = "the escape byte")]
            self.lo.push(ESCAPE as u8);
            put_varint(v - ESCAPE, &mut self.hi);
        }
    }

    fn push_xy(&mut self, r: Xy) {
        self.push(zigzag(r.0));
        self.push(zigzag(r.1));
    }
}

/// The runs of the vertex sequence by where they start, to find the run of a copy's source.
struct RunTable {
    /// First vertex of each run, ascending, then the vertex count.
    starts: Vec<u32>,
    closed: Vec<bool>,
}

impl RunTable {
    fn new(runs: &[Run]) -> Self {
        let mut starts = Vec::with_capacity(runs.len() + 1);
        let mut at = 0u32;
        for &(len, _) in runs {
            starts.push(at);
            at = at.saturating_add(len);
        }
        starts.push(at);
        Self {
            starts,
            closed: runs.iter().map(|&(_, k)| k == RunKind::Ring).collect(),
        }
    }

    /// The (first vertex, length, closed) of the run vertex `k` is in.
    fn run_of(&self, k: usize) -> (usize, usize, bool) {
        // The last start at or before `k`; zero-length runs share a start, and own no vertex.
        let r = self.starts.partition_point(|&s| s.into_usize() <= k) - 1;
        let start = self.starts[r].into_usize();
        (
            start,
            self.starts[r + 1].into_usize() - start,
            self.closed[r],
        )
    }
}

/// Where every vertex already written can be found again, for copies: LZ77-style hash chains.
struct Index {
    runs: RunTable,
    /// The latest written occurrence of each vertex.
    head: rustc_hash::FxHashMap<(i32, i32), u32>,
    /// Per written vertex, the occurrence of the same vertex before it, or `u32::MAX`.
    chain: Vec<u32>,
}

/// The longest copy of the vertices from `i` of the run at `s` out of the `written` vertices:
/// (source vertex, backwards, length).
fn longest_copy(
    v: &[i32],
    s: usize,
    len: usize,
    i: usize,
    written: usize,
    index: &Index,
) -> Option<(usize, bool, usize)> {
    let key = (v[2 * (s + i)], v[2 * (s + i) + 1]);
    let mut j = *index.head.get(&key)?;
    let mut best: Option<(usize, bool, usize)> = None;
    for _ in 0..COPY_CANDIDATES {
        if j == u32::MAX {
            break;
        }
        let at = j.into_usize();
        for backwards in [false, true] {
            let n = copy_len(v, s, len, i, at, backwards, written, index);
            if best.is_none_or(|b| n > b.2) {
                best = Some((at, backwards, n));
            }
        }
        j = index.chain[at];
    }
    best
}

/// How far the vertices from `i` of the run at `s` repeat the written ones from `j` on.
#[expect(clippy::too_many_arguments)]
fn copy_len(
    v: &[i32],
    s: usize,
    len: usize,
    i: usize,
    j: usize,
    backwards: bool,
    written: usize,
    index: &Index,
) -> usize {
    let run = index.runs.run_of(j);
    let mut n = 0;
    let mut src = Some(j);
    while let Some(k) = src {
        if i + n >= len
            || k >= written
            || v[2 * k..2 * k + 2] != v[2 * (s + i + n)..2 * (s + i + n) + 2]
        {
            break;
        }
        n += 1;
        src = step(k, backwards, written, run);
    }
    n
}

/// The vertex a copy reads after `k`: the next or previous one of `run` (its first vertex,
/// length and closedness), wrapping round a closed ring, or [`None`] past the end of the run or
/// of what is written so far. A copy never leaves its source run.
/// The encoder and the decoder both step through this one function, so they cannot disagree.
fn step(k: usize, backwards: bool, written: usize, run: (usize, usize, bool)) -> Option<usize> {
    let (start, len, closed) = run;
    let complete = start + len <= written;
    let last = start + len - 1;
    let next = if backwards {
        if k > start {
            k - 1
        } else if closed && complete {
            last
        } else {
            return None;
        }
    } else if k < last {
        k + 1
    } else if closed && complete {
        start
    } else {
        return None;
    };
    (next < written).then_some(next)
}

/// Encode `vertices`, `(x, y)` words laid out as `runs`, guessing the start of each ring or
/// line from the start or `from_end` the end of the run before it.
#[must_use]
pub fn encode(vertices: &[i32], runs: &[Run], from_end: bool) -> Vec<u8> {
    let mut planes: [Planes; CLASSES] = Default::default();
    let mut ops = Vec::new();
    let mut any_copy = false;
    let mut index = Index {
        runs: RunTable::new(runs),
        head: rustc_hash::FxHashMap::default(),
        chain: Vec::with_capacity(vertices.len() / 2),
    };
    let (mut prev, mut prev_start) = ((0, 0), (0, 0));
    let mut s = 0;
    for &(len, kind) in runs {
        let len = len.into_usize();
        let mut copies: Vec<(usize, usize, bool, usize)> = Vec::new();
        let (mut i, mut literal_from) = (0, 0);
        while i < len {
            if kind != RunKind::Point
                && let Some((j, backwards, n)) = longest_copy(vertices, s, len, i, s + i, &index)
                && n >= MIN_COPY
            {
                copies.push((i - literal_from, s + i - j, backwards, n));
                for k in i..i + n {
                    remember(vertices, s + k, &mut index);
                }
                i += n;
                literal_from = i;
                continue;
            }
            let start_guess = if from_end { prev } else { prev_start };
            let (c, r) = residual(vertices, s, len, kind, i, prev, start_guess);
            planes[c].push_xy(r);
            remember(vertices, s + i, &mut index);
            i += 1;
        }
        if kind != RunKind::Point {
            put_varint(copies.len() as u64, &mut ops);
            any_copy |= !copies.is_empty();
            for (gap, distance, backwards, n) in copies {
                put_varint(gap as u64, &mut ops);
                put_varint((distance as u64) << 1 | u64::from(backwards), &mut ops);
                put_varint((n - MIN_COPY) as u64, &mut ops);
            }
        }
        if len > 0 {
            if kind != RunKind::Point {
                prev_start = at(vertices, s);
            }
            prev = at(vertices, s + len - 1);
        }
        s += len;
    }
    let flags = if any_copy { FLAG_COPIES } else { 0 } | if from_end { FLAG_FROM_END } else { 0 };
    let mut out = vec![flags];
    if any_copy {
        put_varint(ops.len() as u64, &mut out);
        out.extend_from_slice(&ops);
    }
    for p in &planes {
        let mut lo_len = Vec::new();
        put_varint(p.lo.len() as u64, &mut lo_len);
        put_varint((lo_len.len() + p.lo.len() + p.hi.len()) as u64, &mut out);
        out.extend_from_slice(&lo_len);
        out.extend_from_slice(&p.lo);
        out.extend_from_slice(&p.hi);
    }
    out
}

fn remember(v: &[i32], k: usize, index: &mut Index) {
    debug_assert_eq!(index.chain.len(), k, "vertices are remembered in order");
    #[expect(clippy::cast_possible_truncation, reason = "vertex counts fit u32")]
    let previous = index.head.insert((v[2 * k], v[2 * k + 1]), k as u32);
    index.chain.push(previous.unwrap_or(u32::MAX));
}

/// A cursor over one block of the payload.
struct Cursor<'a> {
    data: &'a [u8],
}

impl<'a> Cursor<'a> {
    fn varint(&mut self) -> MltResult<u64> {
        let (rest, v) = parse_varint::<u64>(self.data)?;
        self.data = rest;
        Ok(v)
    }

    fn take(&mut self, n: u64) -> MltResult<&'a [u8]> {
        let n = usize::try_from(n).map_err(|_| MltError::MalformedPredicted("block too long"))?;
        let (head, rest) = self
            .data
            .split_at_checked(n)
            .ok_or(MltError::MalformedPredicted("block past the payload"))?;
        self.data = rest;
        Ok(head)
    }
}

/// One residual stream read back: the zigzag-decoded values of its low and high planes.
fn unpack_plane(lo: &[u8], hi: &[u8]) -> MltResult<Vec<i64>> {
    let mut out = Vec::with_capacity(lo.len());
    let mut at = 0;
    for &b in lo {
        let z = if u64::from(b) < ESCAPE {
            u64::from(b)
        } else {
            escaped(hi, &mut at)?
        };
        out.push(unzigzag(z));
    }
    if at != hi.len() {
        return Err(MltError::MalformedPredicted("trailing escaped residuals"));
    }
    Ok(out)
}

/// The escaped value whose varint starts at `hi[*at]`, which most residuals of a stream of
/// coordinate edges are, so it is read inline rather than through the generic varint reader.
#[inline]
fn escaped(hi: &[u8], at: &mut usize) -> MltResult<u64> {
    let mut v = 0u64;
    let mut shift = 0;
    loop {
        let b = *hi
            .get(*at)
            .ok_or(MltError::MalformedPredicted("an escaped residual ran out"))?;
        *at += 1;
        if shift > 56 {
            return Err(MltError::MalformedPredicted("escaped residual too long"));
        }
        v |= u64::from(b & 0x7f) << shift;
        if b < 0x80 {
            // A varint MUST be canonical, so a multi-byte one ends in a non-zero byte.
            if b == 0 && shift > 0 {
                return Err(MltError::NonCanonicalVarInt);
            }
            break;
        }
        shift += 7;
    }
    v.checked_add(ESCAPE)
        .ok_or(MltError::MalformedPredicted("residual overflow"))
}

/// The next residual pair of a class stream.
#[inline]
fn next_pair(values: &[i64], cursor: &mut usize) -> MltResult<Xy> {
    match values.get(*cursor..*cursor + 2) {
        Some(&[x, y]) => {
            *cursor += 2;
            Ok((x, y))
        }
        _ => Err(MltError::MalformedPredicted("a residual stream ran out")),
    }
}

#[inline]
fn word(v: i64) -> MltResult<i32> {
    i32::try_from(v).map_err(|_| MltError::MalformedPredicted("vertex out of range"))
}

/// Decode `vertex_count` vertices laid out as `runs` into `(x, y)` words.
pub fn decode(
    data: &[u8],
    runs: &[Run],
    vertex_count: usize,
    dec: &mut Decoder,
) -> MltResult<Vec<i32>> {
    let words = vertex_count
        .checked_mul(2)
        .ok_or(MltError::MalformedPredicted("vertex count overflow"))?;
    dec.consume_items::<i32>(words)?;
    let mut input = Cursor { data };
    let (&flags, rest) = data
        .split_first()
        .ok_or(MltError::MalformedPredicted("empty payload"))?;
    input.data = rest;
    if flags & !(FLAG_COPIES | FLAG_FROM_END) != 0 {
        return Err(MltError::MalformedPredicted("reserved flag bits set"));
    }
    let from_end = flags & FLAG_FROM_END != 0;
    let mut ops = Cursor { data: &[] };
    if flags & FLAG_COPIES != 0 {
        let n = input.varint()?;
        ops.data = input.take(n)?;
    }
    let mut residuals: [Vec<i64>; CLASSES] = Default::default();
    for values in &mut residuals {
        let n = input.varint()?;
        let mut block = Cursor {
            data: input.take(n)?,
        };
        let lo_len = block.varint()?;
        let lo = block.take(lo_len)?;
        dec.consume_items::<i64>(lo.len())?;
        *values = unpack_plane(lo, block.data)?;
    }
    if !input.data.is_empty() {
        return Err(MltError::MalformedPredicted("trailing bytes"));
    }
    let total: usize = runs.iter().map(|&(n, _)| n.into_usize()).sum();
    if total != vertex_count {
        return Err(MltError::MalformedPredicted(
            "the runs do not cover the vertices",
        ));
    }

    let flags_copies = flags & FLAG_COPIES != 0;
    let table = if flags_copies {
        Some(RunTable::new(runs))
    } else {
        None
    };
    let mut cursors = [0usize; CLASSES];
    let mut v = vec![0i32; words];
    let (mut prev, mut prev_start): (Xy, Xy) = ((0, 0), (0, 0));
    let mut s = 0;
    for &(len, kind) in runs {
        let len = len.into_usize();
        let mut copies = if flags_copies && kind != RunKind::Point {
            ops.varint()?
        } else {
            0
        };
        // Literal vertices left before the next copy, while one is pending.
        let mut next_copy = if copies > 0 {
            Some(ops.varint()?.into_usize_checked()?)
        } else {
            None
        };
        let mut i = 0;
        while i < len {
            if next_copy == Some(0) {
                let packed = ops.varint()?;
                let distance = (packed >> 1).into_usize_checked()?;
                let backwards = packed & 1 == 1;
                let n = ops
                    .varint()?
                    .into_usize_checked()?
                    .checked_add(MIN_COPY)
                    .ok_or(MltError::MalformedPredicted("copy too long"))?;
                let here = s + i;
                let mut k = here
                    .checked_sub(distance)
                    .filter(|_| distance > 0)
                    .ok_or(MltError::MalformedPredicted("copy before the first vertex"))?;
                if i + n > len {
                    return Err(MltError::MalformedPredicted("copy past the end of its run"));
                }
                let source = table
                    .as_ref()
                    .expect("copies are only read under their flag")
                    .run_of(k);
                for c in 0..n {
                    let written = here + c;
                    if k >= written {
                        return Err(MltError::MalformedPredicted("copy of an unwritten vertex"));
                    }
                    v[2 * written] = v[2 * k];
                    v[2 * written + 1] = v[2 * k + 1];
                    if c + 1 < n {
                        k = step(k, backwards, written + 1, source)
                            .ok_or(MltError::MalformedPredicted("copy past its source run"))?;
                    }
                }
                i += n;
                copies -= 1;
                next_copy = if copies > 0 {
                    Some(ops.varint()?.into_usize_checked()?)
                } else {
                    None
                };
                continue;
            }
            let w = s + i;
            let p = if i == 0 {
                match kind {
                    RunKind::Point => add(prev, next_pair(&residuals[POINT], &mut cursors[POINT])?),
                    RunKind::Line | RunKind::Ring => {
                        let guess = if from_end { prev } else { prev_start };
                        add(guess, next_pair(&residuals[START], &mut cursors[START])?)
                    }
                }
            } else {
                let c = class(kind, len, i);
                let r = next_pair(&residuals[c], &mut cursors[c])?;
                match c {
                    CORNER => add(r, sub(add(at(&v, s), at(&v, s + 2)), at(&v, s + 1))),
                    _ => add(at(&v, w - 1), r),
                }
            };
            v[2 * w] = word(p.0)?;
            v[2 * w + 1] = word(p.1)?;
            i += 1;
            if let Some(gap) = next_copy.as_mut() {
                *gap -= 1;
            }
        }
        if copies > 0 || next_copy.is_some() {
            return Err(MltError::MalformedPredicted(
                "a copy past the end of its run",
            ));
        }
        if kind != RunKind::Point && len > 0 {
            prev_start = at(&v, s);
        }
        if len > 0 {
            prev = at(&v, s + len - 1);
        }
        s += len;
    }
    if !ops.data.is_empty() || cursors.iter().zip(&residuals).any(|(&c, r)| c != r.len()) {
        return Err(MltError::MalformedPredicted("trailing residuals"));
    }
    Ok(v)
}

trait IntoUsizeChecked {
    fn into_usize_checked(self) -> MltResult<usize>;
}

impl IntoUsizeChecked for u64 {
    fn into_usize_checked(self) -> MltResult<usize> {
        usize::try_from(self).map_err(|_| MltError::MalformedPredicted("value too large"))
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    fn round_trip(vertices: &[i32], runs: &[Run]) {
        for from_end in [false, true] {
            let encoded = encode(vertices, runs, from_end);
            let decoded = decode(
                &encoded,
                runs,
                vertices.len() / 2,
                &mut Decoder::with_max_size(u32::MAX),
            )
            .expect("decodes");
            assert_eq!(decoded, vertices, "from_end: {from_end}");
        }
    }

    /// The example of the specification, byte for byte.
    #[test]
    fn decodes_the_example_of_the_specification() {
        let payload = [
            0x00, // flags
            0x05, 0x04, 0x14, 0x14, 0x13, 0x13, // Start
            0x01, 0x00, // Point
            0x05, 0x04, 0x14, 0x00, 0x00, 0x14, // Ring
            0x03, 0x02, 0x00, 0x00, // Corner
            0x03, 0x02, 0x0A, 0x00, // Line1
            0x03, 0x02, 0x0A, 0x02, // Line
        ];
        let runs = [(4, RunKind::Ring), (3, RunKind::Line)];
        let vertices = [10, 10, 20, 10, 20, 20, 10, 20, 0, 0, 5, 0, 10, 1];
        assert_eq!(
            decode(&payload, &runs, 7, &mut Decoder::with_max_size(u32::MAX)).unwrap(),
            vertices
        );
        assert_eq!(encode(&vertices, &runs, false), payload);
    }

    #[test]
    fn rejects_the_reserved_flag_bits() {
        let runs = [(2, RunKind::Line)];
        let mut encoded = encode(&[1, 2, 3, 4], &runs, false);
        for bit in [1, 3, 4, 5, 6, 7] {
            encoded[0] = 1 << bit;
            assert!(decode(&encoded, &runs, 2, &mut Decoder::with_max_size(u32::MAX)).is_err());
        }
    }

    #[test]
    fn a_building_beside_a_shared_wall() {
        // Two squares sharing the wall from (10, 0) to (10, 10), then a road and two points.
        let vertices = [
            0, 0, 10, 0, 10, 10, 0, 10, // ring 1
            10, 10, 10, 0, 20, 0, 20, 10, // ring 2 walks the shared wall backwards
            0, 0, 5, 5, 10, 10, 15, 15, // line
            3, 4, // point
            100, -100, // point
        ];
        let runs = [
            (4, RunKind::Ring),
            (4, RunKind::Ring),
            (4, RunKind::Line),
            (1, RunKind::Point),
            (1, RunKind::Point),
        ];
        round_trip(&vertices, &runs);
        assert_eq!(encode(&vertices, &runs, false)[0], FLAG_COPIES);
    }

    #[test]
    fn chained_segments_start_where_the_last_one_ended() {
        // A road stored as two-vertex segments, each starting where the one before it ended.
        let vertices: Vec<i32> = (0..40)
            .flat_map(|i| {
                [
                    i * 37 % 101,
                    i * 53 % 97,
                    (i + 1) * 37 % 101,
                    (i + 1) * 53 % 97,
                ]
            })
            .collect();
        let runs = vec![(2, RunKind::Line); 40];
        round_trip(&vertices, &runs);
        let from_start = encode(&vertices, &runs, false);
        let from_end = encode(&vertices, &runs, true);
        assert_eq!(from_end[0], FLAG_FROM_END);
        // Every start lands on its guess, a zero byte per coordinate.
        #[expect(clippy::naive_bytecount, reason = "a short test payload")]
        let zeros = |p: &[u8]| p[1..].iter().filter(|&&b| b == 0).count();
        assert!(zeros(&from_end) >= zeros(&from_start) + 2 * 39);
    }

    #[test]
    fn copies_wrap_round_a_closed_ring() {
        let vertices = [0, 0, 10, 0, 10, 10, 0, 10, 10, 0, 0, 0, 0, 10, 5, 20];
        round_trip(&vertices, &[(4, RunKind::Ring), (4, RunKind::Line)]);
    }

    #[test]
    fn empty_runs_keep_the_previous_start() {
        let vertices = [5, 5, 9, 1, 40, 40, 41, 47, 3, 3];
        round_trip(
            &vertices,
            &[
                (2, RunKind::Line),
                (0, RunKind::Ring),
                (2, RunKind::Ring),
                (0, RunKind::Line),
                (1, RunKind::Point),
            ],
        );
    }

    #[test]
    fn rejects_trailing_bytes() {
        let vertices = [1, 2, 3, 4];
        let runs = [(2, RunKind::Line)];
        let mut encoded = encode(&vertices, &runs, false);
        encoded.push(0);
        assert!(decode(&encoded, &runs, 2, &mut Decoder::with_max_size(u32::MAX)).is_err());
    }

    proptest! {
        #[test]
        fn round_trips(
            runs in prop::collection::vec((0u32..12, 0u8..3), 0..30),
            seed in any::<u64>(),
            spread in 1i32..5000,
        ) {
            let runs: Vec<Run> = runs
                .into_iter()
                .map(|(n, k)| match k {
                    0 => (1, RunKind::Point),
                    1 => (n, RunKind::Line),
                    _ => (n, RunKind::Ring),
                })
                .collect();
            let total: usize = runs.iter().map(|r| r.0 as usize).sum();
            // A small grid, so vertices repeat and copies happen.
            let mut state = seed;
            let mut vertices = Vec::with_capacity(total * 2);
            for _ in 0..total * 2 {
                state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                vertices.push(((state >> 33) as i32).rem_euclid(spread) - spread / 2);
            }
            round_trip(&vertices, &runs);
        }

        #[test]
        fn extreme_coordinates_round_trip(
            pts in prop::collection::vec((any::<i32>(), any::<i32>()), 1..20),
        ) {
            let vertices: Vec<i32> = pts.iter().flat_map(|&(x, y)| [x, y]).collect();
            let n = u32::try_from(pts.len()).unwrap();
            round_trip(&vertices, &[(n, RunKind::Line)]);
            round_trip(&vertices, &[(n, RunKind::Ring)]);
        }
    }
}
