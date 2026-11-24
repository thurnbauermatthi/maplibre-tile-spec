//! `glTF` models placed onto `Web Mercator` tiles, simplified across zooms, and encoded as v2 `Meshes` layers.

use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::ops::RangeInclusive;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context as _, Result as AnyResult, bail};
use globset::{GlobBuilder, GlobMatcher};
use martin_tile_utils::EARTH_CIRCUMFERENCE;
use meshopt::{SimplifyOptions, optimize_vertex_fetch, simplify_decoder};
use mlt_core::geo_types::{Geometry, MultiPoint, Point};
use mlt_core::geojson::PropertySchema;
use mlt_core::{HeightMode, MValue, MValueScope, PropKind, PropValue, TileLayer, ZAxis, ZStep};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use serde_json::{Map, Value};

use super::Ground;
use super::geodesy::{self, Frame, MIN_ECEF_RADIUS, Y_UP_TO_Z_UP, srgb};
use super::reader::{Model, Paint, Primitive};
use crate::convert::from_geojson::{mercator_unit, tile_index};
use crate::convert::{Reencoder, tiled};

/// Property naming a landmark or `glTF` file.
pub(super) const NAME: &str = "name";
/// Property holding a landmark's height in metres above its ground datum.
pub(super) const HEIGHT: &str = "height";
/// Triangle m-value naming the architectural surface part (`wall`, `roof`, `ground`, `window`, `door`).
pub(super) const PART: &str = "part";
/// Triangle m-value holding the packed `0xRRGGBB` `sRGB` surface colour.
pub(super) const COLOR: &str = "color";

const UNIT_METRES: f64 = 0.01;
const MAX_EXTENT: u32 = 1 << 21;
const TILE_PIXELS: f64 = 512.0;
const SIMPLIFY_ERROR_PIXELS: f64 = 0.45;
/// `sin(30 deg)`: triangles Tilted at least 30 degrees toward zenith/nadir count as roof/ground when unnamed.
const ROOF_COS: f64 = 0.5;

/// Architectural part labels matching the `Meshes` specification and viewer palette.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Part {
    Wall,
    Roof,
    Ground,
    Window,
    Door,
}

impl Part {
    pub(super) fn parse(name: &str) -> AnyResult<Self> {
        match name {
            "wall" => Ok(Self::Wall),
            "roof" => Ok(Self::Roof),
            "ground" => Ok(Self::Ground),
            "window" => Ok(Self::Window),
            "door" => Ok(Self::Door),
            other => bail!(
                "unknown mesh part {other:?}; expected one of \"wall\", \"roof\", \"ground\", \"window\" or \"door\""
            ),
        }
    }

    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Wall => "wall",
            Self::Roof => "roof",
            Self::Ground => "ground",
            Self::Window => "window",
            Self::Door => "door",
        }
    }

    pub(super) const fn default_color(self) -> u32 {
        match self {
            Self::Wall => 0x00d8_d2c8,
            Self::Roof => 0x00b8_6b52,
            Self::Ground => 0x006e_6a63,
            Self::Window => 0x006d_90a8,
            Self::Door => 0x006b_4c35,
        }
    }

    /// Classifies an unnamed triangle from `dot(normal, up)` and `|normal|`.
    pub(super) fn by_slope(up_dot: f64, norm: f64) -> Self {
        if norm == 0.0 {
            Self::Wall
        } else if up_dot >= ROOF_COS * norm {
            Self::Roof
        } else if up_dot <= -ROOF_COS * norm {
            Self::Ground
        } else {
            Self::Wall
        }
    }
}

/// Ordered glob rules mapping `glTF` node, mesh or material names to [`Part`].
pub(super) struct PartRules {
    rules: Vec<(GlobMatcher, Part)>,
}

impl PartRules {
    /// Compiles explicit `[[parts]]` rules, or builds default case-insensitive architectural rules when empty.
    pub(super) fn new(explicit: &[(String, Part)]) -> AnyResult<Self> {
        if !explicit.is_empty() {
            let rules = explicit
                .iter()
                .map(|(pattern, part)| {
                    let matcher = GlobBuilder::new(pattern)
                        .build()
                        .with_context(|| format!("invalid [[parts]] glob {pattern:?}"))?
                        .compile_matcher();
                    Ok((matcher, *part))
                })
                .collect::<AnyResult<_>>()?;
            return Ok(Self { rules });
        }
        let defaults = [
            ("*roof*", Part::Roof),
            ("*terrace*", Part::Roof),
            ("*metal*", Part::Roof),
            ("*window*", Part::Window),
            ("*glass*", Part::Window),
            ("*door*", Part::Door),
            ("*entrance*", Part::Door),
            ("*portal*", Part::Door),
            ("*ground*", Part::Ground),
            ("*floor*", Part::Ground),
            ("*terrain*", Part::Ground),
            ("*wall*", Part::Wall),
            ("*facade*", Part::Wall),
            ("*column*", Part::Wall),
            ("*tower*", Part::Wall),
            ("*stone*", Part::Wall),
            ("*trim*", Part::Wall),
            ("*terracotta*", Part::Wall),
        ];
        let rules = defaults
            .into_iter()
            .map(|(pattern, part)| {
                let matcher = GlobBuilder::new(pattern)
                    .case_insensitive(true)
                    .build()
                    .context("invalid built-in part glob")?
                    .compile_matcher();
                Ok((matcher, part))
            })
            .collect::<AnyResult<_>>()?;
        Ok(Self { rules })
    }

    fn classify(&self, prim: &Primitive) -> Option<Part> {
        let names = [
            prim.node_name.as_deref(),
            prim.mesh_name.as_deref(),
            prim.material_name.as_deref(),
        ];
        for (matcher, part) in &self.rules {
            if names.into_iter().flatten().any(|n| matcher.is_match(n)) {
                return Some(*part);
            }
        }
        None
    }
}

/// How a `glTF` model is positioned on the `WGS84` ellipsoid.
#[derive(Clone, Copy, Debug)]
pub(super) enum Placement {
    /// Local tangent frame at `[lon_deg, lat_deg]`.
    Anchor([f64; 2]),
    /// Georeferenced in `ECEF` metres via `CESIUM_RTC`.
    Rtc([f64; 3]),
}

/// One zoom-range entry of a [`Landmark`].
pub(super) struct LodModel {
    pub zooms: RangeInclusive<u8>,
    pub placement: Placement,
    pub ground: Ground,
    pub model: Arc<Model>,
}

/// One feature (a single `glTF` file or one `[[landmarks]]` entry) to place across zooms.
pub(super) struct Landmark {
    pub id: u64,
    pub name: Option<String>,
    pub attributes: Map<String, Value>,
    pub models: Vec<LodModel>,
}

/// Shared layer settings across all written tiles.
pub(super) struct LayerSpec {
    pub name: String,
    pub z_step: ZStep,
    pub schema: PropertySchema,
    pub part_rules: PartRules,
}

/// Places `landmarks` onto `zooms`, simplifies lower zooms, and writes v2 `Meshes` tiles under `output`.
pub(super) fn tile(
    landmarks: &[Landmark],
    spec: &LayerSpec,
    zooms: RangeInclusive<u8>,
    output: &Path,
    reencoder: &Reencoder,
) -> AnyResult<usize> {
    let mut written = 0;
    for zoom in zooms {
        let placed: Vec<Option<(Tile, PlacedMesh)>> = landmarks
            .par_iter()
            .map(|landmark| place_landmark(landmark, zoom, spec))
            .collect::<AnyResult<_>>()?;
        let mut by_tile: BTreeMap<Tile, Vec<(&Landmark, PlacedMesh)>> = BTreeMap::new();
        for (landmark, item) in landmarks.iter().zip(placed) {
            if let Some((tile, mesh)) = item {
                by_tile.entry(tile).or_default().push((landmark, mesh));
            }
        }
        written += by_tile
            .into_par_iter()
            .map(|(tile, items)| -> AnyResult<usize> {
                let Some(layer) = build_layer(spec, tile, items)? else {
                    return Ok(0);
                };
                if tiled::write_tile(output, reencoder, tile.zoom, tile.x, tile.y, layer)? {
                    Ok(1)
                } else {
                    eprintln!(
                        "warning: skipping tile {}/{}/{}: exceeds default decoder memory budget",
                        tile.zoom, tile.x, tile.y
                    );
                    Ok(0)
                }
            })
            .try_reduce(|| 0, |a, b| Ok(a + b))?;
    }
    Ok(written)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Tile {
    zoom: u8,
    x: u32,
    y: u32,
}

impl Tile {
    fn of([ux, uy]: [f64; 2], zoom: u8) -> Self {
        let n = 1 << zoom;
        Self {
            zoom,
            x: tile_index(ux, n),
            y: tile_index(uy, n),
        }
    }

    fn width(self) -> f64 {
        let n = f64::from(1_u32 << self.zoom);
        let lat = (PI * (1.0 - 2.0 * (f64::from(self.y) + 0.5) / n))
            .sinh()
            .atan();
        EARTH_CIRCUMFERENCE * lat.cos() / n
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the exponent is clamped to 6..=21"
    )]
    fn extent(self) -> u32 {
        let exponent = (self.width() / UNIT_METRES).log2().floor().clamp(6.0, 21.0) as u32;
        (1 << exponent).min(MAX_EXTENT)
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "the rounded values are checked to fit i32"
    )]
    fn grid(self, [ux, uy]: [f64; 2], extent: u32) -> Option<[i32; 2]> {
        let n = f64::from(1_u32 << self.zoom);
        let extent = f64::from(extent);
        let x = ((ux * n - f64::from(self.x)) * extent).round_ties_even();
        let y = ((uy * n - f64::from(self.y)) * extent).round_ties_even();
        let fits = |v: f64| (f64::from(i32::MIN)..=f64::from(i32::MAX)).contains(&v);
        (fits(x) && fits(y)).then_some([x as i32, y as i32])
    }
}

struct PlacedMesh {
    xyz: Vec<[i32; 3]>,
    triangles: Vec<[u32; 3]>,
    parts: Vec<Part>,
    colors: Vec<u32>,
    height: f64,
}

fn place_landmark(
    landmark: &Landmark,
    zoom: u8,
    spec: &LayerSpec,
) -> AnyResult<Option<(Tile, PlacedMesh)>> {
    let Some(lod) = landmark.models.iter().find(|m| m.zooms.contains(&zoom)) else {
        return Ok(None);
    };
    if lod.model.primitives.is_empty() {
        return Ok(None);
    }

    let mut prim_geo: Vec<Vec<[f64; 3]>> = Vec::with_capacity(lod.model.primitives.len());
    let mut prim_ecef: Vec<Vec<[f64; 3]>> = Vec::new();
    let (mut min_u, mut max_u) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    let (mut min_h, mut max_h) = (f64::INFINITY, f64::NEG_INFINITY);

    match lod.placement {
        Placement::Anchor(anchor) => {
            let frame = Frame::new(anchor);
            for prim in &lod.model.primitives {
                let mut geo = Vec::with_capacity(prim.positions.len());
                for &pos in &prim.positions {
                    let g = frame.to_geodetic(pos);
                    update_bounds(g, &mut min_u, &mut max_u, &mut min_h, &mut max_h);
                    geo.push(g);
                }
                prim_geo.push(geo);
            }
        }
        Placement::Rtc(rtc) => {
            let rtc_mat = geodesy::mul(&geodesy::translation(rtc), &Y_UP_TO_Z_UP);
            for prim in &lod.model.primitives {
                let mut ecef = Vec::with_capacity(prim.positions.len());
                let mut geo = Vec::with_capacity(prim.positions.len());
                for &pos in &prim.positions {
                    let p = geodesy::apply(&rtc_mat, pos);
                    if geodesy::norm(p) <= MIN_ECEF_RADIUS {
                        bail!(
                            "CESIUM_RTC center {rtc:?} does not place landmark {:?} on the Earth",
                            landmark.name.as_deref().unwrap_or("<unnamed>")
                        );
                    }
                    let g = geodesy::ecef_to_geodetic(p);
                    update_bounds(g, &mut min_u, &mut max_u, &mut min_h, &mut max_h);
                    ecef.push(p);
                    geo.push(g);
                }
                prim_ecef.push(ecef);
                prim_geo.push(geo);
            }
        }
    }

    if !min_h.is_finite() {
        return Ok(None);
    }
    let ground_h = match (lod.ground, lod.placement) {
        (Ground::Lowest, _) => min_h,
        (Ground::Origin, Placement::Anchor(_)) => 0.0,
        (Ground::Origin, Placement::Rtc(rtc)) => geodesy::ecef_to_geodetic(rtc)[2],
    };
    let height = ((max_h - ground_h).max(0.0) * 100.0).round() / 100.0;

    let centre = [
        f64::midpoint(min_u[0], max_u[0]),
        f64::midpoint(min_u[1], max_u[1]),
    ];
    let tile = Tile::of(centre, zoom);
    let extent = tile.extent();

    let mut xyz: Vec<[i32; 3]> = Vec::new();
    let mut welded: FxHashMap<[i32; 3], u32> = FxHashMap::default();
    let mut triangles: Vec<[u32; 3]> = Vec::new();
    let mut parts: Vec<Part> = Vec::new();
    let mut colors: Vec<u32> = Vec::new();

    for (prim_idx, prim) in lod.model.primitives.iter().enumerate() {
        let geo = &prim_geo[prim_idx];
        let mut vertex_map: Vec<u32> = Vec::with_capacity(geo.len());
        for &[lon, lat, h] in geo {
            let (ux, uy) = mercator_unit(lon.to_degrees(), lat.to_degrees());
            let Some([x, y]) = tile.grid([ux, uy], extent) else {
                return Ok(None);
            };
            let Some(z) = spec.z_step.z(h - ground_h) else {
                return Ok(None);
            };
            let next_idx = u32::try_from(xyz.len())
                .context("landmark has more welded vertices than a u32 indexes")?;
            let idx = *welded.entry([x, y, z]).or_insert_with(|| {
                xyz.push([x, y, z]);
                next_idx
            });
            vertex_map.push(idx);
        }

        let matched_part = spec.part_rules.classify(prim);
        for (tri_idx, &tri) in prim.triangles.iter().enumerate() {
            let mapped = tri.map(|i| vertex_map[i as usize]);
            if mapped[0] == mapped[1] || mapped[1] == mapped[2] || mapped[0] == mapped[2] {
                continue;
            }
            let part = matched_part.unwrap_or_else(|| match lod.placement {
                Placement::Anchor(_) => {
                    let [a, b, c] = tri.map(|i| prim.positions[i as usize]);
                    let normal = geodesy::cross(geodesy::sub(b, a), geodesy::sub(c, a));
                    Part::by_slope(normal[1], geodesy::norm(normal))
                }
                Placement::Rtc(_) => {
                    let ecef = &prim_ecef[prim_idx];
                    let [a, b, c] = tri.map(|i| ecef[i as usize]);
                    let normal = geodesy::cross(geodesy::sub(b, a), geodesy::sub(c, a));
                    let g = geo[tri[0] as usize];
                    Part::by_slope(
                        geodesy::dot(normal, geodesy::ecef_up([g[0], g[1]])),
                        geodesy::norm(normal),
                    )
                }
            });
            let color = match &prim.paint {
                Paint::Default => part.default_color(),
                Paint::Uniform(rgb) => srgb(*rgb),
                Paint::PerTriangle(per_tri) => srgb(per_tri[tri_idx]),
            };
            triangles.push(mapped);
            parts.push(part);
            colors.push(color);
        }
    }

    if triangles.is_empty() {
        return Ok(None);
    }

    let mut mesh = compact(&xyz, triangles, parts, colors);
    if zoom < *lod.zooms.end() {
        let metres = tile.width() / f64::from(extent);
        let error = SIMPLIFY_ERROR_PIXELS * tile.width() / TILE_PIXELS;
        if let Some(simplified) = simplify(&mesh, [metres, metres, spec.z_step.metres()], error) {
            mesh = simplified;
        }
    }

    if mesh.triangles.is_empty() {
        return Ok(None);
    }
    Ok(Some((
        tile,
        PlacedMesh {
            xyz: mesh.xyz,
            triangles: mesh.triangles,
            parts: mesh.parts,
            colors: mesh.colors,
            height,
        },
    )))
}

fn update_bounds(
    [lon, lat, h]: [f64; 3],
    min_u: &mut [f64; 2],
    max_u: &mut [f64; 2],
    min_h: &mut f64,
    max_h: &mut f64,
) {
    let (ux, uy) = mercator_unit(lon.to_degrees(), lat.to_degrees());
    min_u[0] = min_u[0].min(ux);
    min_u[1] = min_u[1].min(uy);
    max_u[0] = max_u[0].max(ux);
    max_u[1] = max_u[1].max(uy);
    *min_h = min_h.min(h);
    *max_h = max_h.max(h);
}

struct CompactMesh {
    xyz: Vec<[i32; 3]>,
    triangles: Vec<[u32; 3]>,
    parts: Vec<Part>,
    colors: Vec<u32>,
}

/// Simplifies `mesh` within `error` metres via `meshoptimizer` QEM, keeping `(part, color)` seams intact.
fn simplify(mesh: &CompactMesh, metres: [f64; 3], error: f64) -> Option<CompactMesh> {
    let Some(&origin) = mesh.xyz.first() else {
        return Some(compact(&[], Vec::new(), Vec::new(), Vec::new()));
    };
    #[expect(
        clippy::cast_possible_truncation,
        reason = "metres relative to a landmark's first vertex fit an f32"
    )]
    let positions: Vec<[f32; 3]> = mesh
        .xyz
        .iter()
        .map(|p| [0, 1, 2].map(|i| ((f64::from(p[i]) - f64::from(origin[i])) * metres[i]) as f32))
        .collect();
    #[expect(
        clippy::cast_possible_truncation,
        reason = "pixel simplification error in metres fits an f32"
    )]
    let error = error as f32;
    let options =
        SimplifyOptions::ErrorAbsolute | SimplifyOptions::PreserveFolds | SimplifyOptions::Prune;

    // Duplicate vertices across distinct `(part, color)` pairs so material seams simplify along borders without bleeding.
    let mut copies: Vec<Vec<((Part, u32), u32)>> = vec![Vec::new(); mesh.xyz.len()];
    let mut source: Vec<u32> = Vec::new();
    let mut tag_of: Vec<(Part, u32)> = Vec::new();
    let split: Vec<u32> = mesh
        .triangles
        .iter()
        .zip(mesh.parts.iter().copied().zip(mesh.colors.iter().copied()))
        .flat_map(|(t, tag)| t.map(|v| (v, tag)))
        .map(|(v, tag)| {
            let slot = &mut copies[v as usize];
            if let Some(&(_, copy)) = slot.iter().find(|&&(t, _)| t == tag) {
                return Some(copy);
            }
            let copy = u32::try_from(source.len()).ok()?;
            slot.push((tag, copy));
            source.push(v);
            tag_of.push(tag);
            Some(copy)
        })
        .collect::<Option<_>>()?;
    let split_positions: Vec<[f32; 3]> = source.iter().map(|&v| positions[v as usize]).collect();
    let simplified = simplify_decoder(&split, &split_positions, 0, error, options, None);
    let mut triangles = Vec::new();
    let mut parts = Vec::new();
    let mut colors = Vec::new();
    for t in simplified.as_chunks::<3>().0 {
        let mapped = t.map(|copy| source[copy as usize]);
        if mapped[0] == mapped[1] || mapped[1] == mapped[2] || mapped[0] == mapped[2] {
            continue;
        }
        let (part, color) = tag_of[t[0] as usize];
        triangles.push(mapped);
        parts.push(part);
        colors.push(color);
    }
    Some(compact(&mesh.xyz, triangles, parts, colors))
}

fn compact(
    xyz: &[[i32; 3]],
    mut triangles: Vec<[u32; 3]>,
    parts: Vec<Part>,
    colors: Vec<u32>,
) -> CompactMesh {
    let xyz = optimize_vertex_fetch(triangles.as_flattened_mut(), xyz);
    CompactMesh {
        xyz,
        triangles,
        parts,
        colors,
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "clamped to 0..=65535"
)]
fn hilbert_key([x, y]: [i32; 2], extent: u32) -> u32 {
    let span = (i64::from(extent) * 3).max(1);
    let norm =
        |c: i32| (((i64::from(c) + i64::from(extent)).clamp(0, span - 1) * 65535) / span) as u32;
    hilbert_2d::u32::xy2h_discrete(norm(x), norm(y), 16, hilbert_2d::Variant::Hilbert)
}

fn build_layer(
    spec: &LayerSpec,
    tile: Tile,
    mut items: Vec<(&Landmark, PlacedMesh)>,
) -> AnyResult<Option<TileLayer>> {
    items.retain(|(_, mesh)| !mesh.triangles.is_empty());
    if items.is_empty() {
        return Ok(None);
    }
    let extent = tile.extent();
    items.sort_by_key(|(_, mesh)| {
        let [x, y, _] = mesh.xyz.first().copied().unwrap_or_default();
        hilbert_key([x, y], extent)
    });

    let mut builder = TileLayer::builder(spec.name.clone(), extent)?;
    builder.set_meshes(ZAxis {
        step: spec.z_step,
        mode: HeightMode::RelativeToGround,
    })?;
    let has_name = items.iter().any(|(l, _)| l.name.is_some());
    let name_col = has_name
        .then(|| builder.add_property(NAME, PropKind::Str))
        .transpose()?;
    let height_col = builder.add_property(HEIGHT, PropKind::F64)?;
    let attr_cols = spec
        .schema
        .names()
        .iter()
        .zip(spec.schema.kinds())
        .map(|(n, kind)| builder.add_property(n.clone(), *kind))
        .collect::<Result<Vec<_>, _>>()?;
    let part_col = builder.add_scoped_m_value(PART, PropKind::Str, MValueScope::Triangle)?;
    let color_col = builder.add_scoped_m_value(COLOR, PropKind::U32, MValueScope::Triangle)?;

    for (landmark, mesh) in items {
        let points = mesh.xyz.iter().map(|&[x, y, _]| Point::new(x, y)).collect();
        let mut feature = builder.feature(Geometry::MultiPoint(MultiPoint(points)));
        feature.id(Some(landmark.id));
        feature.z(mesh.xyz.iter().map(|v| v[2]).collect())?;
        feature.triangles(mesh.triangles)?;
        if let Some(col) = name_col {
            feature.property(
                col,
                PropValue::Str(landmark.name.as_deref().map(Into::into)),
            )?;
        }
        feature.property(height_col, PropValue::F64(Some(mesh.height)))?;
        for (col_idx, value) in spec.schema.values(&landmark.attributes)? {
            feature.property(attr_cols[col_idx], value)?;
        }
        let parts = mesh.parts.iter().map(|p| p.as_str().into()).collect();
        feature.m_value(part_col, MValue::Str(Some(parts)))?;
        feature.m_value(color_col, MValue::U32(Some(mesh.colors)))?;
        feature.finish()?;
    }
    Ok(Some(builder.finish()))
}
