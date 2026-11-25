//! Tier A: strict lossless comparison of a decoded MLT tile against the ground-truth MVT parse.

use anyhow::{Result, anyhow, bail, ensure};
use mlt_core::{Decoder, Parser};

use crate::canon::{self, CFeature, CLayer};

/// Decode every layer of an MLT tile with `mlt-core` into canonical form.
pub fn decode_mlt(bytes: &[u8]) -> Result<Vec<CLayer>> {
    // The default 64 MB budget is a safety limit for untrusted input; Overture tiles exceed it.
    let mut dec = Decoder::with_max_size(u32::MAX);
    let mut out = Vec::new();
    for layer in Parser::default().parse_layers(bytes)? {
        let tile = layer
            .into_tile(&mut dec)?
            .ok_or_else(|| anyhow!("unknown layer tag"))?;
        out.push(canon::from_tile_layer(&tile)?);
    }
    Ok(out)
}

/// The layers MLT can carry: a layer without features encodes to no bytes.
pub fn non_empty(layers: &[CLayer]) -> Vec<&CLayer> {
    layers.iter().filter(|l| !l.features.is_empty()).collect()
}

fn describe(f: &CFeature) -> String {
    let verts: usize = f.parts.iter().map(Vec::len).sum();
    format!(
        "id={:?} type={} parts={} verts={} props={}",
        f.id,
        f.gtype,
        f.parts.len(),
        verts,
        f.props.len()
    )
}

fn first_diff(e: &CFeature, a: &CFeature) -> String {
    if e.id != a.id {
        return format!("id {:?} vs {:?}", a.id, e.id);
    }
    if e.gtype != a.gtype {
        return format!("geometry type {} vs {}", a.gtype, e.gtype);
    }
    if e.parts != a.parts {
        if e.parts.len() != a.parts.len() {
            return format!("part count {} vs {}", a.parts.len(), e.parts.len());
        }
        for (i, (pe, pa)) in e.parts.iter().zip(&a.parts).enumerate() {
            if pe != pa {
                return format!(
                    "part {i}: {} verts vs {}, first {:?} vs {:?}",
                    pa.len(),
                    pe.len(),
                    pa.first(),
                    pe.first()
                );
            }
        }
    }
    for (pe, pa) in e.props.iter().zip(&a.props) {
        if pe != pa {
            return format!("prop {pa:?} vs {pe:?}");
        }
    }
    format!("prop count {} vs {}", a.props.len(), e.props.len())
}

/// Strict: same non-empty layers in order, same features in order, identical in every field.
pub fn strict(expected: &[CLayer], actual: &[CLayer]) -> Result<()> {
    let expected = non_empty(expected);
    ensure!(
        expected.len() == actual.len(),
        "{} layers, expected {}",
        actual.len(),
        expected.len()
    );
    for (e, a) in expected.iter().zip(actual) {
        ensure!(e.name == a.name, "layer {:?}, expected {:?}", a.name, e.name);
        ensure!(e.extent == a.extent, "layer {}: extent {}", e.name, a.extent);
        ensure!(
            e.features.len() == a.features.len(),
            "layer {}: {} features, expected {}",
            e.name,
            a.features.len(),
            e.features.len()
        );
        for (i, (fe, fa)) in e.features.iter().zip(&a.features).enumerate() {
            if fe != fa {
                bail!(
                    "layer {} feature {i} ({}): {}",
                    e.name,
                    describe(fe),
                    first_diff(fe, fa)
                );
            }
        }
    }
    Ok(())
}
