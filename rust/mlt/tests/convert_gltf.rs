//! End-to-end tests of `mlt convert` tiling `.glb`, `.gltf`, and landmark `.toml` manifests into v2 `Meshes` layers.

#![cfg(not(feature = "hotpath"))]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

use mlt_core::geo_types::Geometry;
use mlt_core::{
    Decoder, HeightMode, MValue, MValueScope, Parser, PropValue, TileLayer, ZAxis, ZStep,
};
use serde_json::{Value, json};

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

struct TempDir(PathBuf);

impl TempDir {
    fn new() -> Self {
        let id = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("mlt-convert-gltf-test-{}-{id}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn write_bytes(&self, rel: &str, bytes: &[u8]) -> PathBuf {
        let path = self.0.join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, bytes).unwrap();
        path
    }

    fn write_str(&self, rel: &str, text: &str) -> PathBuf {
        self.write_bytes(rel, text.as_bytes())
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn mlt_convert(input: &Path, output: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_mlt"))
        .arg("convert")
        .arg(input)
        .arg(output)
        .args(args)
        .env_remove("RUST_BACKTRACE")
        .output()
        .unwrap()
}

#[track_caller]
fn convert(input: &Path, output: &Path, args: &[&str]) {
    let out = mlt_convert(input, output, args);
    assert!(
        out.status.success(),
        "mlt convert failed ({}):\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
}

fn find_tiles(dir: &Path, zoom: u8) -> Vec<(PathBuf, TileLayer)> {
    let zoom_dir = dir.join(zoom.to_string());
    if !zoom_dir.exists() {
        return Vec::new();
    }
    let mut files = Vec::new();
    for x_entry in fs::read_dir(&zoom_dir).unwrap() {
        let x_path = x_entry.unwrap().path();
        if x_path.is_dir() {
            for y_entry in fs::read_dir(&x_path).unwrap() {
                let y_path = y_entry.unwrap().path();
                if y_path.extension().and_then(|s| s.to_str()) == Some("mlt") {
                    files.push(y_path);
                }
            }
        }
    }
    files.sort();
    files
        .into_iter()
        .map(|path| {
            let bytes = fs::read(&path).unwrap();
            let raw = Parser::default()
                .parse_layers(&bytes)
                .unwrap()
                .pop()
                .unwrap();
            let layer = raw.into_tile(&mut Decoder::default()).unwrap().unwrap();
            (path, layer)
        })
        .collect()
}

/// Packs a JSON `glTF` document and binary buffer into a valid version-2 `.glb` byte vector.
fn build_glb(doc: &Value, bin: &[u8]) -> Vec<u8> {
    let mut json_bytes = serde_json::to_vec(doc).unwrap();
    while !json_bytes.len().is_multiple_of(4) {
        json_bytes.push(b' ');
    }
    let mut bin_bytes = bin.to_vec();
    while !bin_bytes.len().is_multiple_of(4) {
        bin_bytes.push(0);
    }
    let total_len = 12 + 8 + json_bytes.len() + 8 + bin_bytes.len();
    let mut out = Vec::with_capacity(total_len);
    out.extend_from_slice(&0x4654_6C67_u32.to_le_bytes());
    out.extend_from_slice(&2_u32.to_le_bytes());
    out.extend_from_slice(&u32::try_from(total_len).unwrap().to_le_bytes());
    out.extend_from_slice(&u32::try_from(json_bytes.len()).unwrap().to_le_bytes());
    out.extend_from_slice(&0x4E4F_534A_u32.to_le_bytes());
    out.extend_from_slice(&json_bytes);
    out.extend_from_slice(&u32::try_from(bin_bytes.len()).unwrap().to_le_bytes());
    out.extend_from_slice(&0x004E_4942_u32.to_le_bytes());
    out.extend_from_slice(&bin_bytes);
    out
}

struct PrimInput<'a> {
    node_name: &'a str,
    material_name: &'a str,
    color: [f32; 4],
    positions: Vec<[f32; 3]>,
    triangles: Vec<[u32; 3]>,
}

/// Builds a multi-primitive `.glb` with optional `CESIUM_RTC` centre.
fn make_primitives_glb(prims: &[PrimInput<'_>], rtc_center: Option<[f64; 3]>) -> Vec<u8> {
    let mut bin: Vec<u8> = Vec::new();
    let mut buffer_views = Vec::new();
    let mut accessors = Vec::new();
    let mut materials = Vec::new();
    let mut meshes = Vec::new();
    let mut nodes = Vec::new();
    let mut scene_nodes = Vec::new();

    for (idx, prim) in prims.iter().enumerate() {
        while !bin.len().is_multiple_of(4) {
            bin.push(0);
        }
        let pos_offset = bin.len();
        for p in &prim.positions {
            for c in *p {
                bin.extend_from_slice(&c.to_le_bytes());
            }
        }
        let pos_len = bin.len() - pos_offset;
        let pos_view = buffer_views.len();
        buffer_views.push(json!({
            "buffer": 0,
            "byteOffset": pos_offset,
            "byteLength": pos_len,
        }));
        let pos_acc = accessors.len();
        accessors.push(json!({
            "bufferView": pos_view,
            "componentType": 5126,
            "count": prim.positions.len(),
            "type": "VEC3",
        }));

        let ind_offset = bin.len();
        for t in &prim.triangles {
            for i in *t {
                bin.extend_from_slice(&i.to_le_bytes());
            }
        }
        let ind_len = bin.len() - ind_offset;
        let ind_view = buffer_views.len();
        buffer_views.push(json!({
            "buffer": 0,
            "byteOffset": ind_offset,
            "byteLength": ind_len,
        }));
        let ind_acc = accessors.len();
        accessors.push(json!({
            "bufferView": ind_view,
            "componentType": 5125,
            "count": prim.triangles.len() * 3,
            "type": "SCALAR",
        }));

        materials.push(json!({
            "name": prim.material_name,
            "pbrMetallicRoughness": {
                "baseColorFactor": prim.color,
            }
        }));
        meshes.push(json!({
            "name": format!("mesh_{idx}"),
            "primitives": [{
                "attributes": { "POSITION": pos_acc },
                "indices": ind_acc,
                "material": idx,
            }]
        }));
        nodes.push(json!({
            "name": prim.node_name,
            "mesh": idx,
        }));
        scene_nodes.push(idx);
    }

    let mut doc = json!({
        "asset": { "version": "2.0" },
        "scene": 0,
        "scenes": [{ "nodes": scene_nodes }],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{ "byteLength": bin.len() }],
    });
    if let Some(center) = rtc_center {
        doc["extensions"] = json!({ "CESIUM_RTC": { "center": center } });
        doc["extensionsUsed"] = json!(["CESIUM_RTC"]);
    }
    build_glb(&doc, &bin)
}

type IndexedMesh = (Vec<[f32; 3]>, Vec<[u32; 3]>);

/// 8-triangle 4-wall prism (`[-half, half]` in X/Z, `[y0, y1]` in Y) + 4-triangle pyramidal roof rising to `apex_y`.
fn tower_with_roof(half: f32, y0: f32, y1: f32, apex_y: f32) -> (IndexedMesh, IndexedMesh) {
    let wall_pos = vec![
        [-half, y0, half],
        [half, y0, half],
        [half, y0, -half],
        [-half, y0, -half],
        [-half, y1, half],
        [half, y1, half],
        [half, y1, -half],
        [-half, y1, -half],
    ];
    let wall_tris = vec![
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ];
    let roof_pos = vec![
        [-half, y1, half],
        [half, y1, half],
        [half, y1, -half],
        [-half, y1, -half],
        [0.0, apex_y, 0.0],
    ];
    let roof_tris = vec![[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]];
    ((wall_pos, wall_tris), (roof_pos, roof_tris))
}

fn encode_base64(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let b0 = u32::from(chunk[0]);
        let b1 = u32::from(*chunk.get(1).unwrap_or(&0));
        let b2 = u32::from(*chunk.get(2).unwrap_or(&0));
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(TABLE[((triple >> 18) & 0x3F) as usize] as char);
        out.push(TABLE[((triple >> 12) & 0x3F) as usize] as char);
        out.push(if chunk.len() > 1 {
            TABLE[((triple >> 6) & 0x3F) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            TABLE[(triple & 0x3F) as usize] as char
        } else {
            '='
        });
    }
    out
}

#[test]
fn a_glb_model_placed_with_anchor_tiles_into_a_meshes_layer_with_part_and_color_m_values() {
    let dir = TempDir::new();
    let ((wall_pos, wall_tris), (roof_pos, roof_tris)) = tower_with_roof(10.0, 0.0, 60.0, 70.0);
    let glb = make_primitives_glb(
        &[
            PrimInput {
                node_name: "Facade_Wall",
                material_name: "Limestone",
                color: [0.8, 0.75, 0.65, 1.0],
                positions: wall_pos,
                triangles: wall_tris,
            },
            PrimInput {
                node_name: "Main_Roof",
                material_name: "Copper",
                color: [0.15, 0.45, 0.35, 1.0],
                positions: roof_pos,
                triangles: roof_tris,
            },
        ],
        None,
    );
    let input = dir.write_bytes("opera.glb", &glb);
    let output = dir.0.join("tiles");
    convert(
        &input,
        &output,
        &[
            "--mlt-version",
            "2",
            "--min-zoom",
            "16",
            "--max-zoom",
            "16",
            "--anchor",
            "2.3316,48.8720",
            "--verify",
        ],
    );

    let tiles = find_tiles(&output, 16);
    assert_eq!(tiles.len(), 1);
    let (_, layer) = &tiles[0];
    assert_eq!(layer.name(), "opera");
    assert!(layer.is_meshes());
    assert_eq!(
        layer.z_axis(),
        Some(ZAxis {
            step: ZStep::new(-2).unwrap(),
            mode: HeightMode::RelativeToGround,
        })
    );
    assert_eq!(layer.features().len(), 1);

    let feature = &layer.features()[0];
    assert_eq!(feature.id(), Some(0));
    assert!(matches!(feature.geometry(), Geometry::MultiPoint(_)));
    assert_eq!(feature.triangles().unwrap().len(), 12);
    assert_eq!(*feature.z().iter().min().unwrap(), 0);
    assert_eq!(*feature.z().iter().max().unwrap(), 7000);

    let props: BTreeMap<&str, &PropValue> = layer
        .property_names()
        .iter()
        .map(String::as_str)
        .zip(feature.properties())
        .collect();
    assert_eq!(
        props.get("name"),
        Some(&&PropValue::Str(Some("opera".into())))
    );
    assert_eq!(props.get("height"), Some(&&PropValue::F64(Some(70.0))));

    let part_idx = layer
        .m_value_names()
        .iter()
        .position(|n| n == "part")
        .unwrap();
    let color_idx = layer
        .m_value_names()
        .iter()
        .position(|n| n == "color")
        .unwrap();
    assert_eq!(layer.m_value_scopes()[part_idx], MValueScope::Triangle);
    assert_eq!(layer.m_value_scopes()[color_idx], MValueScope::Triangle);

    let MValue::Str(Some(parts)) = &feature.m_values()[part_idx] else {
        panic!("expected string part m-values");
    };
    let MValue::U32(Some(colors)) = &feature.m_values()[color_idx] else {
        panic!("expected u32 color m-values");
    };
    assert_eq!(parts.iter().filter(|p| p.as_str() == "wall").count(), 8);
    assert_eq!(parts.iter().filter(|p| p.as_str() == "roof").count(), 4);
    assert_eq!(colors.len(), 12);
    // Wall and roof carry distinct sRGB colours derived from their linear PBR factors.
    let wall_color = colors[parts.iter().position(|p| p.as_str() == "wall").unwrap()];
    let roof_color = colors[parts.iter().position(|p| p.as_str() == "roof").unwrap()];
    assert_ne!(wall_color, roof_color);
}

#[test]
fn lower_zooms_auto_simplify_a_single_glb_while_preserving_part_and_color_seams() {
    let dir = TempDir::new();
    let radial = 64_u16;
    let rings = 8_u16;
    let radius = 25.0_f32;
    let mut wall_pos = Vec::new();
    let mut wall_tris = Vec::new();
    for ring in 0..=rings {
        let y = (f32::from(ring) / f32::from(rings)) * 40.0;
        for seg in 0..radial {
            let angle = (f32::from(seg) / f32::from(radial)) * std::f32::consts::TAU;
            wall_pos.push([radius * angle.cos(), y, -radius * angle.sin()]);
        }
    }
    for ring in 0..u32::from(rings) {
        let rad = u32::from(radial);
        for seg in 0..rad {
            let next_s = (seg + 1) % rad;
            let idx_a = ring * rad + seg;
            let idx_b = ring * rad + next_s;
            let idx_c = (ring + 1) * rad + next_s;
            let idx_d = (ring + 1) * rad + seg;
            wall_tris.push([idx_a, idx_b, idx_c]);
            wall_tris.push([idx_a, idx_c, idx_d]);
        }
    }

    let mut roof_pos = Vec::new();
    let mut roof_tris = Vec::new();
    for ring in 0..=rings {
        let frac = (f32::from(ring) / f32::from(rings)) * std::f32::consts::FRAC_PI_2;
        let ring_r = radius * frac.cos();
        let y = 40.0 + 20.0 * frac.sin();
        for seg in 0..radial {
            let angle = (f32::from(seg) / f32::from(radial)) * std::f32::consts::TAU;
            roof_pos.push([ring_r * angle.cos(), y, -ring_r * angle.sin()]);
        }
    }
    for ring in 0..u32::from(rings) {
        let rad = u32::from(radial);
        for seg in 0..rad {
            let next_s = (seg + 1) % rad;
            let idx_a = ring * rad + seg;
            let idx_b = ring * rad + next_s;
            let idx_c = (ring + 1) * rad + next_s;
            let idx_d = (ring + 1) * rad + seg;
            roof_tris.push([idx_a, idx_b, idx_c]);
            roof_tris.push([idx_a, idx_c, idx_d]);
        }
    }

    let glb = make_primitives_glb(
        &[
            PrimInput {
                node_name: "Rotunda_Wall",
                material_name: "Stone",
                color: [0.75, 0.72, 0.68, 1.0],
                positions: wall_pos,
                triangles: wall_tris,
            },
            PrimInput {
                node_name: "Dome_Roof",
                material_name: "Lead",
                color: [0.22, 0.25, 0.30, 1.0],
                positions: roof_pos,
                triangles: roof_tris,
            },
        ],
        None,
    );
    let input = dir.write_bytes("pantheon.glb", &glb);
    let output = dir.0.join("tiles");
    convert(
        &input,
        &output,
        &[
            "--mlt-version",
            "2",
            "--min-zoom",
            "13",
            "--max-zoom",
            "16",
            "--anchor",
            "2.3462,48.8462",
            "--verify",
        ],
    );

    let counts: Vec<usize> = (13..=16)
        .map(|z| {
            let tiles = find_tiles(&output, z);
            assert_eq!(tiles.len(), 1, "expected 1 tile at zoom {z}");
            let layer = &tiles[0].1;
            let feat = &layer.features()[0];
            let part_idx = layer
                .m_value_names()
                .iter()
                .position(|n| n == "part")
                .unwrap();
            let MValue::Str(Some(parts)) = &feat.m_values()[part_idx] else {
                panic!("expected string parts");
            };
            assert!(
                parts.iter().any(|p| p.as_str() == "wall"),
                "missing wall at z{z}"
            );
            assert!(
                parts.iter().any(|p| p.as_str() == "roof"),
                "missing roof at z{z}"
            );
            feat.triangles().unwrap().len()
        })
        .collect();

    // z16 keeps the full mesh; z15, z14 and z13 progressively simplify.
    assert!(
        counts[3] > counts[2] && counts[2] > counts[1] && counts[1] > counts[0] && counts[0] > 0,
        "expected strictly increasing triangle counts from z13 to z16, got {counts:?}"
    );
}

#[test]
fn a_gltf_json_with_external_bin_base64_uri_mirrored_trs_and_vertex_colors_tiles_cleanly() {
    let dir = TempDir::new();
    // Horizontal roof quad at y = 12 m (unindexed, 2 triangles = 6 vertices) + vertical wall quad (indexed, mirrored X scale)
    // with normalized UNSIGNED_BYTE VEC4 COLOR_0.
    let roof_positions: [[f32; 3]; 6] = [
        [-8.0, 12.0, 8.0],
        [8.0, 12.0, 8.0],
        [8.0, 12.0, -8.0],
        [-8.0, 12.0, 8.0],
        [8.0, 12.0, -8.0],
        [-8.0, 12.0, -8.0],
    ];
    let mut ext_bin = Vec::new();
    for p in roof_positions {
        for c in p {
            ext_bin.extend_from_slice(&c.to_le_bytes());
        }
    }
    dir.write_bytes("roof geom.bin", &ext_bin);

    // Wall quad in +X, mirrored via scale [-1, 1, 1], with red VEC4 u8 vertex colors and u16 indices.
    let wall_positions: [[f32; 3]; 4] = [
        [8.0, 0.0, 8.0],
        [8.0, 12.0, 8.0],
        [8.0, 12.0, -8.0],
        [8.0, 0.0, -8.0],
    ];
    let wall_colors: [[u8; 4]; 4] = [[255, 0, 0, 255]; 4];
    // Under [-1, 1, 1], CCW winding in local space flips unless the reader reverses mirrored triangles.
    let wall_indices: [u16; 6] = [0, 1, 2, 0, 2, 3];
    let mut inline_bin = Vec::new();
    for p in wall_positions {
        for c in p {
            inline_bin.extend_from_slice(&c.to_le_bytes());
        }
    }
    let color_offset = inline_bin.len();
    for rgba in wall_colors {
        inline_bin.extend_from_slice(&rgba);
    }
    let ind_offset = inline_bin.len();
    for i in wall_indices {
        inline_bin.extend_from_slice(&i.to_le_bytes());
    }
    let data_uri = format!(
        "data:application/octet-stream;base64,{}",
        encode_base64(&inline_bin)
    );

    let gltf_doc = json!({
        "asset": { "version": "2.0" },
        "scene": 0,
        "scenes": [{ "nodes": [0, 1] }],
        "nodes": [
            { "mesh": 0 },
            { "mesh": 1, "scale": [-1.0, 1.0, 1.0], "translation": [0.0, 0.0, 0.0] }
        ],
        "meshes": [
            {
                "primitives": [{
                    "attributes": { "POSITION": 0 }
                }]
            },
            {
                "primitives": [{
                    "attributes": { "POSITION": 1, "COLOR_0": 2 },
                    "indices": 3
                }]
            }
        ],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": 6, "type": "VEC3" },
            { "bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC3" },
            { "bufferView": 2, "componentType": 5121, "normalized": true, "count": 4, "type": "VEC4" },
            { "bufferView": 3, "componentType": 5123, "count": 6, "type": "SCALAR" }
        ],
        "bufferViews": [
            { "buffer": 0, "byteOffset": 0, "byteLength": ext_bin.len() },
            { "buffer": 1, "byteOffset": 0, "byteLength": color_offset },
            { "buffer": 1, "byteOffset": color_offset, "byteLength": 16 },
            { "buffer": 1, "byteOffset": ind_offset, "byteLength": 12 }
        ],
        "buffers": [
            { "uri": "roof%20geom.bin", "byteLength": ext_bin.len() },
            { "uri": data_uri, "byteLength": inline_bin.len() }
        ]
    });
    let input = dir.write_str(
        "pavilion.gltf",
        &serde_json::to_string_pretty(&gltf_doc).unwrap(),
    );
    let output = dir.0.join("tiles");
    convert(
        &input,
        &output,
        &[
            "--mlt-version",
            "2",
            "--max-zoom",
            "16",
            "--min-zoom",
            "16",
            "--anchor",
            "9.9937,53.5511",
            "--verify",
        ],
    );

    let tiles = find_tiles(&output, 16);
    assert_eq!(tiles.len(), 1);
    let layer = &tiles[0].1;
    let feat = &layer.features()[0];
    assert_eq!(feat.triangles().unwrap().len(), 4);

    let part_idx = layer
        .m_value_names()
        .iter()
        .position(|n| n == "part")
        .unwrap();
    let color_idx = layer
        .m_value_names()
        .iter()
        .position(|n| n == "color")
        .unwrap();
    let MValue::Str(Some(parts)) = &feat.m_values()[part_idx] else {
        panic!("expected string parts");
    };
    let MValue::U32(Some(colors)) = &feat.m_values()[color_idx] else {
        panic!("expected u32 colors");
    };
    // Unnamed horizontal +Y normal -> roof; unnamed vertical normal -> wall with sRGB pure red 0xFF0000.
    assert_eq!(parts.iter().filter(|p| p.as_str() == "roof").count(), 2);
    assert_eq!(parts.iter().filter(|p| p.as_str() == "wall").count(), 2);
    assert!(colors.contains(&0x00FF_0000));
}

#[test]
fn a_cesium_rtc_georeferenced_glb_tiles_without_anchor_and_grounds_at_lowest_vertex() {
    let dir = TempDir::new();
    // Build a tower standing upright along the local WGS84 zenith at Paris (~2.33 E, 48.87 N).
    let (lon, lat): (f64, f64) = (2.3316_f64.to_radians(), 48.8720_f64.to_radians());
    let (slon, clon) = lon.sin_cos();
    let (slat, clat) = lat.sin_cos();
    let east = [-slon, clon, 0.0];
    let north = [-slat * clon, -slat * slon, clat];
    let up = [clat * clon, clat * slon, slat];
    let a = 6_378_137.0_f64;
    let e2 = 0.006_694_379_990_141_316_5_f64;
    let n_rad = a / (1.0 - e2 * slat * slat).sqrt();
    let h0 = 80.0_f64;
    let rtc = [
        (n_rad + h0) * clat * clon,
        (n_rad + h0) * clat * slon,
        (n_rad * (1.0 - e2) + h0) * slat,
    ];
    // Convert local [east, up, south] offsets to glTF Y-up RTC offsets [dx_ecef, dz_ecef, -dy_ecef].
    let to_rtc_yup = |p: [f32; 3]| -> [f32; 3] {
        let (pe, pu, ps) = (f64::from(p[0]), f64::from(p[1]), f64::from(p[2]));
        let pn = -ps;
        let dx = pe * east[0] + pn * north[0] + pu * up[0];
        let dy = pe * east[1] + pn * north[1] + pu * up[1];
        let dz = pe * east[2] + pn * north[2] + pu * up[2];
        #[expect(
            clippy::cast_possible_truncation,
            reason = "local offsets in metres fit f32"
        )]
        [dx as f32, dz as f32, (-dy) as f32]
    };
    let ((wall_pos, wall_tris), (roof_pos, roof_tris)) = tower_with_roof(8.0, -15.0, 25.0, 35.0);
    let glb = make_primitives_glb(
        &[
            PrimInput {
                node_name: "Wall",
                material_name: "Stone",
                color: [0.7, 0.7, 0.7, 1.0],
                positions: wall_pos.into_iter().map(to_rtc_yup).collect(),
                triangles: wall_tris,
            },
            PrimInput {
                node_name: "Roof",
                material_name: "Slate",
                color: [0.2, 0.2, 0.25, 1.0],
                positions: roof_pos.into_iter().map(to_rtc_yup).collect(),
                triangles: roof_tris,
            },
        ],
        Some(rtc),
    );
    let input = dir.write_bytes("rtc_tower.glb", &glb);
    let output = dir.0.join("tiles");
    convert(
        &input,
        &output,
        &[
            "--mlt-version",
            "2",
            "--min-zoom",
            "16",
            "--max-zoom",
            "16",
            "--verify",
        ],
    );

    let tiles = find_tiles(&output, 16);
    assert_eq!(tiles.len(), 1);
    let feat = &tiles[0].1.features()[0];
    let z_min = *feat.z().iter().min().unwrap();
    let z_max = *feat.z().iter().max().unwrap();
    assert_eq!(z_min, 0);
    assert!(
        (4950..=5050).contains(&z_max),
        "expected ~50 m height, got z_max={z_max}"
    );
}

#[test]
fn a_toml_manifest_supports_per_zoom_models_part_globs_negative_longitudes_and_properties() {
    let dir = TempDir::new();
    let ((wall_pos, wall_tris), (roof_pos, roof_tris)) = tower_with_roof(12.0, 0.0, 45.0, 55.0);
    let coarse_glb = make_primitives_glb(
        &[
            PrimInput {
                node_name: "Pedestal_Block",
                material_name: "Granite",
                color: [0.6, 0.6, 0.6, 1.0],
                positions: wall_pos.clone(),
                triangles: wall_tris.clone(),
            },
            PrimInput {
                node_name: "Crown_Glass",
                material_name: "Glass",
                color: [0.3, 0.7, 0.6, 1.0],
                positions: roof_pos.clone(),
                triangles: roof_tris.clone(),
            },
        ],
        None,
    );
    // Fine model adds an extra portal quad (+2 triangles = 14 triangles total vs 12 in coarse).
    let mut fine_prims = vec![
        PrimInput {
            node_name: "Pedestal_Block",
            material_name: "Granite",
            color: [0.6, 0.6, 0.6, 1.0],
            positions: wall_pos,
            triangles: wall_tris,
        },
        PrimInput {
            node_name: "Crown_Glass",
            material_name: "Glass",
            color: [0.3, 0.7, 0.6, 1.0],
            positions: roof_pos,
            triangles: roof_tris,
        },
    ];
    fine_prims.push(PrimInput {
        node_name: "South_Portal",
        material_name: "Bronze",
        color: [0.25, 0.18, 0.10, 1.0],
        positions: vec![
            [-2.0, 0.0, 12.1],
            [2.0, 0.0, 12.1],
            [2.0, 6.0, 12.1],
            [-2.0, 6.0, 12.1],
        ],
        triangles: vec![[0, 1, 2], [0, 2, 3]],
    });
    let fine_glb = make_primitives_glb(&fine_prims, None);

    dir.write_bytes("models/liberty_lod1.glb", &coarse_glb);
    dir.write_bytes("models/liberty_lod2.glb", &fine_glb);

    let manifest = dir.write_str(
        "landmarks.toml",
        r#"
layer = "monuments"
attribution = "Test Landmarks"
ground = "origin"

[[parts]]
match = "*Glass*"
part = "window"

[[parts]]
nodes = ["*Portal*"]
part = "door"

[[parts]]
match = "*"
part = "wall"

[[landmarks]]
id = 42
name = "Statue of Liberty"
anchor = [-74.0445, 40.6892]
models = [
    { gltf = "models/liberty_lod1.glb", min-zoom = 14, max-zoom = 15 },
    { gltf = "models/liberty_lod2.glb", min-zoom = 16, max-zoom = 16 },
]
[landmarks.properties]
architect = "Bartholdi"
opened = 1886
"#,
    );

    let output = dir.0.join("tiles");
    convert(&manifest, &output, &["--mlt-version", "2", "--verify"]);

    let z15 = find_tiles(&output, 15);
    let z16 = find_tiles(&output, 16);
    assert_eq!(z15.len(), 1);
    assert_eq!(z16.len(), 1);
    assert_eq!(z15[0].1.name(), "monuments");
    assert_eq!(z15[0].1.features()[0].triangles().unwrap().len(), 12);
    assert_eq!(z16[0].1.features()[0].triangles().unwrap().len(), 14);

    let layer16 = &z16[0].1;
    let feat16 = &layer16.features()[0];
    assert_eq!(feat16.id(), Some(42));
    let props: BTreeMap<&str, &PropValue> = layer16
        .property_names()
        .iter()
        .map(String::as_str)
        .zip(feat16.properties())
        .collect();
    assert_eq!(
        props.get("name"),
        Some(&&PropValue::Str(Some("Statue of Liberty".into())))
    );
    assert_eq!(
        props.get("architect"),
        Some(&&PropValue::Str(Some("Bartholdi".into())))
    );
    assert_eq!(props.get("opened"), Some(&&PropValue::U32(Some(1886))));

    let part_idx = layer16
        .m_value_names()
        .iter()
        .position(|n| n == "part")
        .unwrap();
    let MValue::Str(Some(parts)) = &feat16.m_values()[part_idx] else {
        panic!("expected string parts");
    };
    assert_eq!(parts.iter().filter(|p| p.as_str() == "window").count(), 4);
    assert_eq!(parts.iter().filter(|p| p.as_str() == "door").count(), 2);
    assert_eq!(parts.iter().filter(|p| p.as_str() == "wall").count(), 8);
}

#[test]
fn gltf_inputs_reject_missing_anchors_overlapping_zooms_and_incompatible_flags() {
    let dir = TempDir::new();
    let ((w_pos, w_tris), (r_pos, r_tris)) = tower_with_roof(5.0, 0.0, 10.0, 14.0);
    let glb = make_primitives_glb(
        &[
            PrimInput {
                node_name: "Wall",
                material_name: "Stone",
                color: [0.8, 0.8, 0.8, 1.0],
                positions: w_pos,
                triangles: w_tris,
            },
            PrimInput {
                node_name: "Roof",
                material_name: "Tile",
                color: [0.5, 0.2, 0.1, 1.0],
                positions: r_pos,
                triangles: r_tris,
            },
        ],
        None,
    );
    let model_path = dir.write_bytes("tower.glb", &glb);
    let geojson_path = dir.write_str(
        "pts.geojson",
        r#"{"type":"FeatureCollection","features":[]}"#,
    );
    let overlap_toml = dir.write_str(
        "overlap.toml",
        r#"
[[landmarks]]
name = "Overlap"
anchor = [2.33, 48.87]
models = [
    { gltf = "tower.glb", min-zoom = 14, max-zoom = 15 },
    { gltf = "tower.glb", min-zoom = 15, max-zoom = 16 },
]
"#,
    );
    let traversal_gltf = dir.write_str(
        "sub/escape.gltf",
        r#"{
            "asset": { "version": "2.0" },
            "buffers": [{ "uri": "../secret.bin", "byteLength": 12 }]
        }"#,
    );

    let cases: &[(&Path, &[&str], &str)] = &[
        (
            &model_path,
            &["--mlt-version", "2", "--max-zoom", "16"],
            "has no CESIUM_RTC extension; pass --anchor",
        ),
        (
            &model_path,
            &["--max-zoom", "16", "--anchor", "2.33,48.87"],
            "tiling a glTF input writes MLT version 2",
        ),
        (
            &model_path,
            &["--mlt-version", "2", "--anchor", "2.33,48.87"],
            "tiling a glTF input requires --max-zoom",
        ),
        (
            &model_path,
            &[
                "--mlt-version",
                "2",
                "--max-zoom",
                "16",
                "--anchor",
                "2.33,48.87",
                "--tessellate",
            ],
            "--tessellate cuts 2D polygons, so it does not apply to a glTF mesh input",
        ),
        (
            &geojson_path,
            &[
                "--mlt-version",
                "2",
                "--max-zoom",
                "4",
                "--anchor",
                "2.33,48.87",
            ],
            "--anchor tiles a glTF input",
        ),
        (
            &overlap_toml,
            &["--mlt-version", "2"],
            "overlapping models at zoom 15",
        ),
        (
            &traversal_gltf,
            &[
                "--mlt-version",
                "2",
                "--max-zoom",
                "16",
                "--anchor",
                "2.33,48.87",
            ],
            "escapes model directory",
        ),
    ];

    for (idx, (input, args, expected)) in cases.iter().enumerate() {
        let out_dir = dir.0.join(format!("fail_{idx}"));
        let out = mlt_convert(input, &out_dir, args);
        assert!(
            !out.status.success(),
            "case {idx} ({expected}) unexpectedly succeeded"
        );
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            stderr.contains(expected),
            "case {idx} stderr did not contain {expected:?}:\n{stderr}"
        );
    }
}
