//! Canonical, interpretation-free tile model used as ground truth for every lossless check.
//!
//! The MVT side is parsed by `mvt.rs`, a protobuf reader written from the spec that shares no
//! code with `fast-mvt` or `mlt-core`. The MLT side is produced by decoding with `mlt-core`.

use std::collections::BTreeMap;

use mlt_core::geo_types::{Coord, Geometry, LineString, Polygon};
use mlt_core::{PropValue, TileLayer};

/// A property value, keyed by its data-model type: MVT `int`/`sint`/`uint` are one integer type,
/// `float`/`double` one float type compared by the exact `f64` bits of the value.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CVal {
    Bool(bool),
    Int(i128),
    Float(u64),
    Str(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CFeature {
    pub id: Option<u64>,
    /// 1 point, 2 line, 3 polygon (MVT `GeomType`).
    pub gtype: u8,
    /// Point parts hold one vertex each; polygon rings exclude the closing vertex.
    pub parts: Vec<Vec<(i32, i32)>>,
    /// Sorted by key; a key appears at most once.
    pub props: Vec<(String, CVal)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CLayer {
    pub name: String,
    pub extent: u32,
    pub features: Vec<CFeature>,
}

fn xy(c: &Coord<i32>) -> (i32, i32) {
    (c.x, c.y)
}

fn line(l: &LineString<i32>) -> Vec<(i32, i32)> {
    l.0.iter().map(xy).collect()
}

/// A decoded ring always repeats its first vertex at the end; drop exactly that one.
fn ring(l: &LineString<i32>) -> Vec<(i32, i32)> {
    let mut v = line(l);
    if v.len() > 1 && v.first() == v.last() {
        v.pop();
    }
    v
}

fn polygon_rings(p: &Polygon<i32>, out: &mut Vec<Vec<(i32, i32)>>) {
    if p.exterior().0.is_empty() && p.interiors().is_empty() {
        return;
    }
    out.push(ring(p.exterior()));
    for r in p.interiors() {
        out.push(ring(r));
    }
}

pub fn geometry(g: &Geometry<i32>) -> anyhow::Result<(u8, Vec<Vec<(i32, i32)>>)> {
    Ok(match g {
        Geometry::Point(p) => (1, vec![vec![xy(&p.0)]]),
        Geometry::MultiPoint(mp) => (1, mp.0.iter().map(|p| vec![xy(&p.0)]).collect()),
        Geometry::LineString(l) => (2, vec![line(l)]),
        Geometry::MultiLineString(ml) => (2, ml.0.iter().map(line).collect()),
        Geometry::Polygon(p) => {
            let mut parts = Vec::new();
            polygon_rings(p, &mut parts);
            (3, parts)
        }
        Geometry::MultiPolygon(mp) => {
            let mut parts = Vec::new();
            for p in &mp.0 {
                polygon_rings(p, &mut parts);
            }
            (3, parts)
        }
        other => anyhow::bail!("unexpected decoded geometry {other:?}"),
    })
}

pub fn value(v: &PropValue) -> Option<CVal> {
    Some(match v {
        PropValue::Bool(v) => CVal::Bool((*v)?),
        PropValue::I8(v) => CVal::Int((*v)?.into()),
        PropValue::U8(v) => CVal::Int((*v)?.into()),
        PropValue::I32(v) => CVal::Int((*v)?.into()),
        PropValue::U32(v) => CVal::Int((*v)?.into()),
        PropValue::I64(v) => CVal::Int((*v)?.into()),
        PropValue::U64(v) => CVal::Int((*v)?.into()),
        PropValue::F32(v) => CVal::Float(f64::from((*v)?).to_bits()),
        PropValue::F64(v) => CVal::Float((*v)?.to_bits()),
        PropValue::Str(v) => CVal::Str(v.as_deref()?.to_string()),
    })
}

/// A decoded MLT layer in canonical form, features in stored order.
pub fn from_tile_layer(layer: &TileLayer) -> anyhow::Result<CLayer> {
    let names = layer.property_names();
    let mut features = Vec::with_capacity(layer.features().len());
    for f in layer.features() {
        let (gtype, parts) = geometry(f.geometry())?;
        let mut props: BTreeMap<String, CVal> = BTreeMap::new();
        for (name, v) in names.iter().zip(f.properties()) {
            if let Some(v) = value(v) {
                anyhow::ensure!(
                    props.insert(name.clone(), v).is_none(),
                    "duplicate column {name}"
                );
            }
        }
        features.push(CFeature {
            id: f.id(),
            gtype,
            parts,
            props: props.into_iter().collect(),
        });
    }
    Ok(CLayer {
        name: layer.name().to_string(),
        extent: layer.extent().get(),
        features,
    })
}
