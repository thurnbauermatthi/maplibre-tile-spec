//! Minimal MVT 2.1 protobuf reader written from the spec, independent of `fast-mvt` and `mlt-core`.

use std::collections::BTreeMap;

use anyhow::{Result, bail, ensure};

use crate::canon::{CFeature, CLayer, CVal};

/// Property keys found more than once in one feature, across every tile parsed.
pub static DUPLICATE_KEYS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

struct Pb<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Pb<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self { b, i: 0 }
    }
    fn done(&self) -> bool {
        self.i >= self.b.len()
    }
    fn varint(&mut self) -> Result<u64> {
        let mut v = 0u64;
        let mut shift = 0;
        loop {
            ensure!(self.i < self.b.len(), "truncated varint");
            let x = self.b[self.i];
            self.i += 1;
            ensure!(shift < 64, "varint too long");
            v |= u64::from(x & 0x7f) << shift;
            if x & 0x80 == 0 {
                return Ok(v);
            }
            shift += 7;
        }
    }
    fn bytes(&mut self) -> Result<&'a [u8]> {
        let n = usize::try_from(self.varint()?)?;
        ensure!(self.i + n <= self.b.len(), "truncated bytes");
        let s = &self.b[self.i..self.i + n];
        self.i += n;
        Ok(s)
    }
    fn fixed<const N: usize>(&mut self) -> Result<[u8; N]> {
        ensure!(self.i + N <= self.b.len(), "truncated fixed");
        let mut a = [0u8; N];
        a.copy_from_slice(&self.b[self.i..self.i + N]);
        self.i += N;
        Ok(a)
    }
    fn key(&mut self) -> Result<(u64, u8)> {
        let k = self.varint()?;
        Ok((k >> 3, (k & 7) as u8))
    }
    fn skip(&mut self, wt: u8) -> Result<()> {
        match wt {
            0 => {
                self.varint()?;
            }
            1 => {
                self.fixed::<8>()?;
            }
            2 => {
                self.bytes()?;
            }
            5 => {
                self.fixed::<4>()?;
            }
            _ => bail!("unsupported wire type {wt}"),
        }
        Ok(())
    }
    /// A repeated uint32 field, packed or not.
    fn repeated_u32(&mut self, wt: u8, out: &mut Vec<u32>) -> Result<()> {
        if wt == 2 {
            let mut p = Pb::new(self.bytes()?);
            while !p.done() {
                out.push(u32::try_from(p.varint()?)?);
            }
        } else {
            ensure!(wt == 0, "bad wire type for repeated uint32");
            out.push(u32::try_from(self.varint()?)?);
        }
        Ok(())
    }
}

fn zz32(v: u32) -> i32 {
    ((v >> 1) as i32) ^ -((v & 1) as i32)
}

fn parse_value(b: &[u8]) -> Result<Option<CVal>> {
    let mut p = Pb::new(b);
    let mut val = None;
    while !p.done() {
        let (f, wt) = p.key()?;
        val = Some(match (f, wt) {
            (1, 2) => CVal::Str(String::from_utf8(p.bytes()?.to_vec())?),
            (2, 5) => CVal::Float(f64::from(f32::from_le_bytes(p.fixed::<4>()?)).to_bits()),
            (3, 1) => CVal::Float(f64::from_le_bytes(p.fixed::<8>()?).to_bits()),
            (4, 0) => CVal::Int(i128::from(p.varint()? as i64)),
            (5, 0) => CVal::Int(i128::from(p.varint()?)),
            (6, 0) => {
                let v = p.varint()?;
                CVal::Int(i128::from(((v >> 1) as i64) ^ -((v & 1) as i64)))
            }
            (7, 0) => CVal::Bool(p.varint()? != 0),
            _ => {
                p.skip(wt)?;
                continue;
            }
        });
    }
    Ok(val)
}

fn decode_geometry(cmds: &[u32]) -> Result<Vec<Vec<(i32, i32)>>> {
    let (mut x, mut y) = (0i32, 0i32);
    let mut parts: Vec<Vec<(i32, i32)>> = Vec::new();
    let mut i = 0;
    while i < cmds.len() {
        let cmd = cmds[i] & 7;
        let count = (cmds[i] >> 3) as usize;
        i += 1;
        match cmd {
            1 | 2 => {
                ensure!(i + 2 * count <= cmds.len(), "truncated geometry");
                for _ in 0..count {
                    x = x.wrapping_add(zz32(cmds[i]));
                    y = y.wrapping_add(zz32(cmds[i + 1]));
                    i += 2;
                    if cmd == 1 {
                        parts.push(vec![(x, y)]);
                    } else {
                        let Some(cur) = parts.last_mut() else {
                            bail!("LineTo before MoveTo")
                        };
                        cur.push((x, y));
                    }
                }
            }
            7 => ensure!(count == 1, "ClosePath count {count}"),
            _ => bail!("bad command {cmd}"),
        }
    }
    Ok(parts)
}

fn parse_feature(b: &[u8], keys: &[String], values: &[Option<CVal>]) -> Result<CFeature> {
    let mut p = Pb::new(b);
    let mut id = None;
    let mut tags = Vec::new();
    let mut gtype = 0u8;
    let mut geom = Vec::new();
    while !p.done() {
        let (f, wt) = p.key()?;
        match (f, wt) {
            (1, 0) => id = Some(p.varint()?),
            (2, _) => p.repeated_u32(wt, &mut tags)?,
            (3, 0) => gtype = u8::try_from(p.varint()?)?,
            (4, _) => p.repeated_u32(wt, &mut geom)?,
            _ => p.skip(wt)?,
        }
    }
    ensure!(tags.len() % 2 == 0, "odd tag count");
    ensure!((1..=3).contains(&gtype), "unsupported geometry type {gtype}");
    let mut props = BTreeMap::new();
    for kv in tags.chunks(2) {
        let k = keys.get(kv[0] as usize).ok_or_else(|| anyhow::anyhow!("bad key index"))?;
        let v = values.get(kv[1] as usize).ok_or_else(|| anyhow::anyhow!("bad value index"))?;
        // A key repeated within a feature keeps its last value, as GL JS's MVT reader and
        // mlt-core's import read it. Such tiles break the MVT data model, so they are counted.
        if let Some(v) = v
            && props.insert(k.clone(), v.clone()).is_some()
        {
            DUPLICATE_KEYS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
    Ok(CFeature {
        id,
        gtype,
        parts: decode_geometry(&geom)?,
        props: props.into_iter().collect(),
    })
}

fn parse_layer(b: &[u8]) -> Result<CLayer> {
    let mut p = Pb::new(b);
    let mut name = String::new();
    let mut extent = 4096u32;
    let mut keys = Vec::new();
    let mut values = Vec::new();
    let mut feats = Vec::new();
    while !p.done() {
        let (f, wt) = p.key()?;
        match (f, wt) {
            (1, 2) => name = String::from_utf8(p.bytes()?.to_vec())?,
            (2, 2) => feats.push(p.bytes()?),
            (3, 2) => keys.push(String::from_utf8(p.bytes()?.to_vec())?),
            (4, 2) => values.push(parse_value(p.bytes()?)?),
            (5, 0) => extent = u32::try_from(p.varint()?)?,
            _ => p.skip(wt)?,
        }
    }
    let features = feats
        .into_iter()
        .map(|f| parse_feature(f, &keys, &values))
        .collect::<Result<Vec<_>>>()?;
    Ok(CLayer {
        name,
        extent,
        features,
    })
}

/// Every layer of an uncompressed MVT tile, in stored order.
pub fn parse_tile(b: &[u8]) -> Result<Vec<CLayer>> {
    let mut p = Pb::new(b);
    let mut layers = Vec::new();
    while !p.done() {
        let (f, wt) = p.key()?;
        if (f, wt) == (3, 2) {
            layers.push(parse_layer(p.bytes()?)?);
        } else {
            p.skip(wt)?;
        }
    }
    Ok(layers)
}
