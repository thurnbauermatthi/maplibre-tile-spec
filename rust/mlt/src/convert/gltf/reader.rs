//! Pure-Rust `glTF` 2.0 `.gltf` and `.glb` reader extracting scene-transformed triangle primitives.

use std::collections::HashSet;
use std::fs;
use std::path::Path;

use anyhow::{Context as _, Result as AnyResult, bail, ensure};
use serde::Deserialize;

use super::geodesy::{self, IDENTITY, Mat4};

const GLB_MAGIC: u32 = 0x4654_6C67;
const GLB_VERSION: u32 = 2;
const CHUNK_JSON: u32 = 0x4E4F_534A;
const CHUNK_BIN: u32 = 0x004E_4942;

const UNSIGNED_BYTE: u32 = 5121;
const UNSIGNED_SHORT: u32 = 5123;
const UNSIGNED_INT: u32 = 5125;
const FLOAT: u32 = 5126;

const MODE_TRIANGLES: u32 = 4;
const MODE_TRIANGLE_STRIP: u32 = 5;
const MODE_TRIANGLE_FAN: u32 = 6;

const UNSUPPORTED_EXTENSIONS: &[&str] = &[
    "KHR_draco_mesh_compression",
    "EXT_meshopt_compression",
    "KHR_mesh_quantization",
];

/// All scene primitives read from one `.gltf` or `.glb` file.
#[derive(Clone, Debug)]
pub(super) struct Model {
    pub rtc_center: Option<[f64; 3]>,
    pub primitives: Vec<Primitive>,
}

/// One mesh primitive transformed into the `glTF` scene's world space (`Y`-up, metres).
#[derive(Clone, Debug)]
pub(super) struct Primitive {
    pub node_name: Option<String>,
    pub mesh_name: Option<String>,
    pub material_name: Option<String>,
    /// Scene-space `[x, y, z]` positions (`+X` east, `+Y` up, `+Z` south for anchored models).
    pub positions: Vec<[f64; 3]>,
    /// Counter-clockwise outward triangle corner indices into `positions`.
    pub triangles: Vec<[u32; 3]>,
    /// Per-triangle colouring resolved from `COLOR_0` and `pbrMetallicRoughness`.
    pub paint: Paint,
}

/// How the triangles of a [`Primitive`] are coloured.
#[derive(Clone, Debug)]
pub(super) enum Paint {
    /// No untextured material colour or `COLOR_0`; fall back to the classified `Part`'s default colour.
    Default,
    /// Uniform linear `[r, g, b]` across every triangle of the primitive.
    Uniform([f32; 3]),
    /// Per-triangle linear `[r, g, b]`, one entry per triangle in `Primitive::triangles`.
    PerTriangle(Vec<[f32; 3]>),
}

/// Reads a `.glb` or `.gltf` file from `path`.
pub(super) fn read(path: &Path) -> AnyResult<Model> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let base_dir = path.parent().unwrap_or_else(|| Path::new("."));
    parse(&bytes, Some(base_dir)).with_context(|| format!("parsing {}", path.display()))
}

/// Parses `.glb` or `.gltf` bytes, resolving relative buffer URIs under `base_dir`.
pub(super) fn parse(bytes: &[u8], base_dir: Option<&Path>) -> AnyResult<Model> {
    let (doc, glb_bin) = if bytes.len() >= 12
        && bytes
            .get(..4)
            .and_then(|b| <[u8; 4]>::try_from(b).ok())
            .map(u32::from_le_bytes)
            == Some(GLB_MAGIC)
    {
        parse_glb(bytes)?
    } else {
        let doc: Document = serde_json::from_slice(bytes).context("invalid glTF 2.0 JSON")?;
        (doc, None)
    };

    for ext in &doc.extensions_required {
        if UNSUPPORTED_EXTENSIONS.contains(&ext.as_str()) {
            bail!(
                "glTF requires unsupported geometry extension {ext}; decompress the model to plain glTF 2.0 first"
            );
        }
    }

    let buffers = resolve_buffers(&doc.buffers, glb_bin, base_dir)?;
    let mut primitives = Vec::new();
    let roots: Vec<usize> =
        if let Some(scene_idx) = doc.scene.or_else(|| (!doc.scenes.is_empty()).then_some(0)) {
            doc.scenes
                .get(scene_idx)
                .context("glTF scene index out of bounds")?
                .nodes
                .clone()
        } else {
            root_nodes(&doc.nodes)
        };

    let mut ancestors = HashSet::new();
    for root in roots {
        visit_node(
            &doc,
            &buffers,
            root,
            &IDENTITY,
            &mut ancestors,
            &mut primitives,
        )?;
    }

    Ok(Model {
        rtc_center: doc.extensions.and_then(|e| e.cesium_rtc).map(|r| r.center),
        primitives,
    })
}

fn parse_glb(bytes: &[u8]) -> AnyResult<(Document, Option<&[u8]>)> {
    let header = bytes.get(..12).context("truncated GLB header")?;
    let version = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
    ensure!(
        version == GLB_VERSION,
        "unsupported GLB version {version} (expected 2)"
    );
    let declared_len = u32::from_le_bytes([header[8], header[9], header[10], header[11]]) as usize;
    let total = declared_len.min(bytes.len());
    ensure!(total >= 12, "truncated GLB container");

    let mut cursor = 12;
    let mut json_chunk: Option<&[u8]> = None;
    let mut bin_chunk: Option<&[u8]> = None;
    while cursor + 8 <= total {
        let hdr = &bytes[cursor..cursor + 8];
        let len = u32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]) as usize;
        let kind = u32::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]);
        cursor += 8;
        let data = bytes
            .get(cursor..cursor.saturating_add(len))
            .context("GLB chunk extends past end of file")?;
        cursor += len;
        match kind {
            CHUNK_JSON if json_chunk.is_none() => json_chunk = Some(data),
            CHUNK_BIN if bin_chunk.is_none() => bin_chunk = Some(data),
            _ => {}
        }
    }
    let json = json_chunk.context("GLB container has no JSON chunk")?;
    let doc: Document = serde_json::from_slice(json).context("invalid JSON chunk in GLB")?;
    Ok((doc, bin_chunk))
}

fn resolve_buffers<'a>(
    specs: &[BufferSpec],
    glb_bin: Option<&'a [u8]>,
    base_dir: Option<&Path>,
) -> AnyResult<Vec<BufferBytes<'a>>> {
    specs
        .iter()
        .enumerate()
        .map(|(idx, spec)| {
            let data = match &spec.uri {
                None => {
                    ensure!(
                        idx == 0,
                        "glTF buffer {idx} has no URI, which only buffer 0 of a GLB may omit"
                    );
                    BufferBytes::Borrowed(
                        glb_bin.context("glTF buffer 0 has no URI and the GLB has no BIN chunk")?,
                    )
                }
                Some(uri) if uri.starts_with("data:") => BufferBytes::Owned(decode_data_uri(uri)?),
                Some(uri) if uri.starts_with("http://") || uri.starts_with("https://") => {
                    bail!("remote glTF buffer URIs are not supported: {uri}")
                }
                Some(uri) => {
                    let dir = base_dir
                        .context("cannot resolve relative glTF buffer URI without a file path")?;
                    let rel = percent_decode(uri)?;
                    let path = safe_join(dir, &rel)?;
                    let raw = fs::read(&path).with_context(|| {
                        format!("reading glTF buffer {idx} at {}", path.display())
                    })?;
                    BufferBytes::Owned(raw)
                }
            };
            ensure!(
                data.as_slice().len() >= spec.byte_length,
                "glTF buffer {idx} is shorter ({} B) than its declared byteLength ({} B)",
                data.as_slice().len(),
                spec.byte_length
            );
            Ok(data)
        })
        .collect()
}

fn root_nodes(nodes: &[Node]) -> Vec<usize> {
    let mut child_set = HashSet::new();
    for node in nodes {
        child_set.extend(node.children.iter().copied());
    }
    (0..nodes.len())
        .filter(|i| !child_set.contains(i))
        .collect()
}

fn visit_node(
    doc: &Document,
    buffers: &[BufferBytes<'_>],
    node_idx: usize,
    parent: &Mat4,
    ancestors: &mut HashSet<usize>,
    out: &mut Vec<Primitive>,
) -> AnyResult<()> {
    ensure!(
        ancestors.insert(node_idx),
        "cycle in glTF scene node hierarchy at node {node_idx}"
    );
    let node = doc
        .nodes
        .get(node_idx)
        .with_context(|| format!("glTF node index {node_idx} out of bounds"))?;
    let local = node.matrix.unwrap_or_else(|| {
        geodesy::from_trs(
            node.translation.unwrap_or([0.0; 3]),
            node.rotation.unwrap_or([0.0, 0.0, 0.0, 1.0]),
            node.scale.unwrap_or([1.0; 3]),
        )
    });
    let world = geodesy::mul(parent, &local);
    let mirrored = geodesy::det3(&world) < 0.0;

    if let Some(mesh_idx) = node.mesh {
        let mesh = doc
            .meshes
            .get(mesh_idx)
            .with_context(|| format!("glTF mesh index {mesh_idx} out of bounds"))?;
        for prim in &mesh.primitives {
            if let Some(decoded) = decode_primitive(
                doc,
                buffers,
                prim,
                &world,
                mirrored,
                node.name.as_deref(),
                mesh.name.as_deref(),
            )? {
                out.push(decoded);
            }
        }
    }
    for &child in &node.children {
        visit_node(doc, buffers, child, &world, ancestors, out)?;
    }
    ancestors.remove(&node_idx);
    Ok(())
}

fn decode_primitive(
    doc: &Document,
    buffers: &[BufferBytes<'_>],
    prim: &PrimitiveSpec,
    world: &Mat4,
    mirrored: bool,
    node_name: Option<&str>,
    mesh_name: Option<&str>,
) -> AnyResult<Option<Primitive>> {
    let mode = prim.mode.unwrap_or(MODE_TRIANGLES);
    if !matches!(
        mode,
        MODE_TRIANGLES | MODE_TRIANGLE_STRIP | MODE_TRIANGLE_FAN
    ) {
        return Ok(None);
    }
    if prim
        .extensions
        .as_ref()
        .is_some_and(|e| e.draco.is_some() || e.meshopt.is_some())
    {
        bail!(
            "glTF primitive uses Draco or meshopt geometry compression; decompress the model to plain glTF 2.0 first"
        );
    }
    let Some(pos_idx) = prim.attributes.position else {
        return Ok(None);
    };
    let raw_positions = read_positions(doc, buffers, pos_idx)?;
    if raw_positions.is_empty() {
        return Ok(None);
    }
    let positions: Vec<[f64; 3]> = raw_positions
        .into_iter()
        .map(|p| geodesy::apply(world, p.map(f64::from)))
        .collect();

    let vertex_count = u32::try_from(positions.len())
        .context("glTF primitive has more vertices than a u32 indexes")?;
    let indices = match prim.indices {
        Some(idx) => read_indices(doc, buffers, idx)?,
        None => (0..vertex_count).collect(),
    };
    let mut triangles = assemble_triangles(&indices, mode, mirrored);
    if triangles.is_empty() {
        return Ok(None);
    }
    for &[a, b, c] in &triangles {
        ensure!(
            a < vertex_count && b < vertex_count && c < vertex_count,
            "glTF triangle vertex index out of bounds ({vertex_count} vertices)"
        );
    }

    let material = match prim.material {
        Some(i) => Some(
            doc.materials
                .get(i)
                .with_context(|| format!("glTF material index {i} out of bounds"))?,
        ),
        None => None,
    };
    let material_name = material.and_then(|m| m.name.clone());
    let pbr = material.and_then(|m| m.pbr_metallic_roughness.as_ref());
    let factor = pbr
        .and_then(|p| p.base_color_factor)
        .map(|[r, g, b, _]| [r, g, b]);
    let textured = pbr.is_some_and(|p| p.base_color_texture.is_some());

    let vertex_colors = match prim.attributes.color_0 {
        Some(idx) => Some(read_colors(doc, buffers, idx, positions.len())?),
        None => None,
    };

    let paint = match (vertex_colors, factor, material.is_some(), textured) {
        (Some(vc), factor, _, _) => {
            let tint = factor.unwrap_or([1.0, 1.0, 1.0]);
            let colors = triangles
                .iter()
                .map(|&[a, b, c]| {
                    let (ca, cb, cc) = (vc[a as usize], vc[b as usize], vc[c as usize]);
                    [0, 1, 2].map(|k| ((ca[k] + cb[k] + cc[k]) / 3.0) * tint[k])
                })
                .collect();
            Paint::PerTriangle(colors)
        }
        (None, _, false, _) | (None, None, true, true) => Paint::Default,
        (None, Some(rgb), true, true) if geodesy::srgb(rgb) == 0x00FF_FFFF => Paint::Default,
        (None, factor, true, _) => Paint::Uniform(factor.unwrap_or([1.0, 1.0, 1.0])),
    };

    triangles.shrink_to_fit();
    Ok(Some(Primitive {
        node_name: node_name.map(str::to_owned),
        mesh_name: mesh_name.map(str::to_owned),
        material_name,
        positions,
        triangles,
        paint,
    }))
}

fn assemble_triangles(indices: &[u32], mode: u32, mirrored: bool) -> Vec<[u32; 3]> {
    let orient = |[a, b, c]: [u32; 3]| if mirrored { [a, c, b] } else { [a, b, c] };
    match mode {
        MODE_TRIANGLES => indices
            .as_chunks::<3>()
            .0
            .iter()
            .map(|&t| orient(t))
            .collect(),
        MODE_TRIANGLE_STRIP => indices
            .windows(3)
            .enumerate()
            .filter(|(_, w)| w[0] != w[1] && w[1] != w[2] && w[0] != w[2])
            .map(|(i, w)| {
                let tri = if i % 2 == 0 {
                    [w[0], w[1], w[2]]
                } else {
                    [w[1], w[0], w[2]]
                };
                orient(tri)
            })
            .collect(),
        MODE_TRIANGLE_FAN => {
            let Some(&first) = indices.first() else {
                return Vec::new();
            };
            indices
                .get(1..)
                .unwrap_or_default()
                .windows(2)
                .filter(|w| first != w[0] && w[0] != w[1] && first != w[1])
                .map(|w| orient([first, w[0], w[1]]))
                .collect()
        }
        _ => Vec::new(),
    }
}

struct SliceView<'a> {
    bytes: &'a [u8],
    stride: usize,
    count: usize,
    component_type: u32,
    accessor_type: &'a str,
    normalized: bool,
}

fn accessor_view<'a>(
    doc: &'a Document,
    buffers: &'a [BufferBytes<'_>],
    idx: usize,
) -> AnyResult<SliceView<'a>> {
    let acc = doc
        .accessors
        .get(idx)
        .with_context(|| format!("glTF accessor {idx} out of bounds"))?;
    ensure!(
        acc.sparse.is_none(),
        "sparse glTF accessors (accessor {idx}) are not supported"
    );
    let view_idx = acc
        .buffer_view
        .with_context(|| format!("glTF accessor {idx} has no bufferView"))?;
    let view = doc
        .buffer_views
        .get(view_idx)
        .with_context(|| format!("glTF bufferView {view_idx} out of bounds"))?;
    let buf = buffers
        .get(view.buffer)
        .with_context(|| format!("glTF buffer {} out of bounds", view.buffer))?
        .as_slice();

    let elem_size = element_size(acc.component_type, &acc.accessor_type)?;
    let stride = view.byte_stride.unwrap_or(elem_size);
    ensure!(
        stride >= elem_size,
        "glTF bufferView {view_idx} byteStride ({stride}) is smaller than element size ({elem_size})"
    );
    let view_end = view
        .byte_offset
        .checked_add(view.byte_length)
        .context("glTF bufferView offset overflow")?;
    let view_bytes = buf
        .get(view.byte_offset..view_end)
        .with_context(|| format!("glTF bufferView {view_idx} extends past buffer"))?;
    let bytes = if acc.count == 0 {
        &[][..]
    } else {
        let needed = (acc.count - 1)
            .checked_mul(stride)
            .and_then(|n| n.checked_add(elem_size))
            .context("glTF accessor byte span overflow")?;
        let end = acc
            .byte_offset
            .checked_add(needed)
            .context("glTF accessor offset overflow")?;
        view_bytes
            .get(acc.byte_offset..end)
            .with_context(|| format!("glTF accessor {idx} extends past bufferView {view_idx}"))?
    };
    Ok(SliceView {
        bytes,
        stride,
        count: acc.count,
        component_type: acc.component_type,
        accessor_type: &acc.accessor_type,
        normalized: acc.normalized,
    })
}

fn element_size(component_type: u32, accessor_type: &str) -> AnyResult<usize> {
    let comp = match component_type {
        5120 | UNSIGNED_BYTE => 1,
        5122 | UNSIGNED_SHORT => 2,
        UNSIGNED_INT | FLOAT => 4,
        other => bail!("unsupported glTF accessor componentType {other}"),
    };
    let arity = match accessor_type {
        "SCALAR" => 1,
        "VEC2" => 2,
        "VEC3" => 3,
        "VEC4" | "MAT2" => 4,
        "MAT3" => 9,
        "MAT4" => 16,
        other => bail!("unsupported glTF accessor type {other}"),
    };
    Ok(comp * arity)
}

fn read_positions(
    doc: &Document,
    buffers: &[BufferBytes<'_>],
    idx: usize,
) -> AnyResult<Vec<[f32; 3]>> {
    let view = accessor_view(doc, buffers, idx)?;
    ensure!(
        view.component_type == FLOAT && view.accessor_type == "VEC3",
        "glTF POSITION accessor {idx} must be VEC3 FLOAT (got {} {})",
        view.accessor_type,
        view.component_type
    );
    let mut out = Vec::with_capacity(view.count);
    for i in 0..view.count {
        let s = &view.bytes[i * view.stride..i * view.stride + 12];
        let p = [
            f32::from_le_bytes([s[0], s[1], s[2], s[3]]),
            f32::from_le_bytes([s[4], s[5], s[6], s[7]]),
            f32::from_le_bytes([s[8], s[9], s[10], s[11]]),
        ];
        ensure!(
            p.iter().all(|v| v.is_finite()),
            "glTF POSITION accessor {idx} contains non-finite coordinate"
        );
        out.push(p);
    }
    Ok(out)
}

fn read_indices(doc: &Document, buffers: &[BufferBytes<'_>], idx: usize) -> AnyResult<Vec<u32>> {
    let view = accessor_view(doc, buffers, idx)?;
    ensure!(
        view.accessor_type == "SCALAR",
        "glTF index accessor {idx} must be SCALAR (got {})",
        view.accessor_type
    );
    let mut out = Vec::with_capacity(view.count);
    for i in 0..view.count {
        let at = i * view.stride;
        let v = match view.component_type {
            UNSIGNED_BYTE => u32::from(view.bytes[at]),
            UNSIGNED_SHORT => u32::from(u16::from_le_bytes([view.bytes[at], view.bytes[at + 1]])),
            UNSIGNED_INT => u32::from_le_bytes([
                view.bytes[at],
                view.bytes[at + 1],
                view.bytes[at + 2],
                view.bytes[at + 3],
            ]),
            other => bail!("unsupported glTF index componentType {other}"),
        };
        out.push(v);
    }
    Ok(out)
}

fn read_colors(
    doc: &Document,
    buffers: &[BufferBytes<'_>],
    idx: usize,
    expected: usize,
) -> AnyResult<Vec<[f32; 3]>> {
    let view = accessor_view(doc, buffers, idx)?;
    ensure!(
        matches!(view.accessor_type, "VEC3" | "VEC4"),
        "glTF COLOR_0 accessor {idx} must be VEC3 or VEC4 (got {})",
        view.accessor_type
    );
    ensure!(
        view.count == expected,
        "glTF COLOR_0 accessor {idx} has {} values, expected {expected}",
        view.count
    );
    let mut out = Vec::with_capacity(view.count);
    for i in 0..view.count {
        let at = i * view.stride;
        let rgb = match view.component_type {
            FLOAT => [
                f32::from_le_bytes([
                    view.bytes[at],
                    view.bytes[at + 1],
                    view.bytes[at + 2],
                    view.bytes[at + 3],
                ]),
                f32::from_le_bytes([
                    view.bytes[at + 4],
                    view.bytes[at + 5],
                    view.bytes[at + 6],
                    view.bytes[at + 7],
                ]),
                f32::from_le_bytes([
                    view.bytes[at + 8],
                    view.bytes[at + 9],
                    view.bytes[at + 10],
                    view.bytes[at + 11],
                ]),
            ],
            UNSIGNED_BYTE if view.normalized => [
                f32::from(view.bytes[at]) / 255.0,
                f32::from(view.bytes[at + 1]) / 255.0,
                f32::from(view.bytes[at + 2]) / 255.0,
            ],
            UNSIGNED_SHORT if view.normalized => [
                f32::from(u16::from_le_bytes([view.bytes[at], view.bytes[at + 1]])) / 65_535.0,
                f32::from(u16::from_le_bytes([view.bytes[at + 2], view.bytes[at + 3]])) / 65_535.0,
                f32::from(u16::from_le_bytes([view.bytes[at + 4], view.bytes[at + 5]])) / 65_535.0,
            ],
            other => bail!(
                "unsupported glTF COLOR_0 componentType {other} (normalized={})",
                view.normalized
            ),
        };
        out.push(rgb);
    }
    Ok(out)
}

enum BufferBytes<'a> {
    Borrowed(&'a [u8]),
    Owned(Vec<u8>),
}

impl BufferBytes<'_> {
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Borrowed(s) => s,
            Self::Owned(v) => v,
        }
    }
}

fn decode_data_uri(uri: &str) -> AnyResult<Vec<u8>> {
    let (meta, payload) = uri
        .split_once(',')
        .context("malformed data: URI in glTF buffer")?;
    ensure!(
        meta.ends_with(";base64"),
        "glTF buffer data: URI must be base64-encoded: {meta}"
    );
    decode_base64(payload)
}

fn decode_base64(input: &str) -> AnyResult<Vec<u8>> {
    let clean: Vec<u8> = input.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    if clean.is_empty() {
        return Ok(Vec::new());
    }
    ensure!(
        clean.len().is_multiple_of(4),
        "invalid base64 length in glTF data: URI"
    );
    let sextet = |b: u8| -> AnyResult<u32> {
        Ok(match b {
            b'A'..=b'Z' => u32::from(b - b'A'),
            b'a'..=b'z' => u32::from(b - b'a' + 26),
            b'0'..=b'9' => u32::from(b - b'0' + 52),
            b'+' => 62,
            b'/' => 63,
            _ => bail!("invalid base64 byte 0x{b:02x} in glTF data: URI"),
        })
    };
    let chunks = clean.as_chunks::<4>().0;
    let mut out = Vec::with_capacity(chunks.len() * 3);
    for (chunk_idx, chunk) in chunks.iter().enumerate() {
        let is_last = chunk_idx + 1 == chunks.len();
        let pad = match (chunk[2] == b'=', chunk[3] == b'=') {
            (true, true) if is_last => 2,
            (false, true) if is_last => 1,
            (false, false) => 0,
            _ => bail!("invalid base64 padding in glTF data: URI"),
        };
        let a = sextet(chunk[0])?;
        let b = sextet(chunk[1])?;
        let c = if pad >= 2 { 0 } else { sextet(chunk[2])? };
        let d = if pad >= 1 { 0 } else { sextet(chunk[3])? };
        let [_, b0, b1, b2] = ((a << 18) | (b << 12) | (c << 6) | d).to_be_bytes();
        out.push(b0);
        if pad < 2 {
            out.push(b1);
        }
        if pad < 1 {
            out.push(b2);
        }
    }
    Ok(out)
}

fn percent_decode(uri: &str) -> AnyResult<String> {
    let bytes = uri.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = uri
                .get(i + 1..i + 3)
                .context("truncated percent-escape in glTF buffer URI")?;
            let byte = u8::from_str_radix(hex, 16)
                .with_context(|| format!("invalid percent-escape %{hex} in glTF buffer URI"))?;
            out.push(byte);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).context("glTF buffer URI is not valid UTF-8")
}

/// Joins a relative `glTF` buffer URI onto `base_dir`, rejecting path traversal outside `base_dir`.
fn safe_join(base_dir: &Path, rel: &str) -> AnyResult<std::path::PathBuf> {
    let rel_path = Path::new(rel);
    ensure!(
        !rel_path.is_absolute(),
        "glTF buffer URI must be relative, got: {rel}"
    );
    let mut depth = 0usize;
    for comp in rel_path.components() {
        match comp {
            std::path::Component::Normal(_) => depth += 1,
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                depth = depth
                    .checked_sub(1)
                    .with_context(|| format!("glTF buffer URI escapes model directory: {rel}"))?;
            }
            std::path::Component::Prefix(_) | std::path::Component::RootDir => {
                bail!("glTF buffer URI must be relative, got: {rel}");
            }
        }
    }
    Ok(base_dir.join(rel_path))
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Document {
    #[serde(default)]
    extensions_required: Vec<String>,
    extensions: Option<RootExtensions>,
    scene: Option<usize>,
    #[serde(default)]
    scenes: Vec<Scene>,
    #[serde(default)]
    nodes: Vec<Node>,
    #[serde(default)]
    meshes: Vec<MeshSpec>,
    #[serde(default)]
    materials: Vec<MaterialSpec>,
    #[serde(default)]
    accessors: Vec<AccessorSpec>,
    #[serde(default)]
    buffer_views: Vec<BufferViewSpec>,
    #[serde(default)]
    buffers: Vec<BufferSpec>,
}

#[derive(Deserialize)]
struct RootExtensions {
    #[serde(rename = "CESIUM_RTC")]
    cesium_rtc: Option<CesiumRtc>,
}

#[derive(Deserialize)]
struct CesiumRtc {
    center: [f64; 3],
}

#[derive(Deserialize)]
struct Scene {
    #[serde(default)]
    nodes: Vec<usize>,
}

#[derive(Deserialize)]
struct Node {
    name: Option<String>,
    mesh: Option<usize>,
    #[serde(default)]
    children: Vec<usize>,
    matrix: Option<[f64; 16]>,
    translation: Option<[f64; 3]>,
    rotation: Option<[f64; 4]>,
    scale: Option<[f64; 3]>,
}

#[derive(Deserialize)]
struct MeshSpec {
    name: Option<String>,
    #[serde(default)]
    primitives: Vec<PrimitiveSpec>,
}

#[derive(Deserialize)]
struct PrimitiveSpec {
    attributes: AttributesSpec,
    indices: Option<usize>,
    material: Option<usize>,
    mode: Option<u32>,
    extensions: Option<PrimitiveExtensions>,
}

#[derive(Deserialize)]
struct AttributesSpec {
    #[serde(rename = "POSITION")]
    position: Option<usize>,
    #[serde(rename = "COLOR_0")]
    color_0: Option<usize>,
}

#[derive(Deserialize)]
struct PrimitiveExtensions {
    #[serde(rename = "KHR_draco_mesh_compression")]
    draco: Option<serde_json::Value>,
    #[serde(rename = "EXT_meshopt_compression")]
    meshopt: Option<serde_json::Value>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct MaterialSpec {
    name: Option<String>,
    pbr_metallic_roughness: Option<PbrSpec>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PbrSpec {
    base_color_factor: Option<[f32; 4]>,
    base_color_texture: Option<serde_json::Value>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct AccessorSpec {
    buffer_view: Option<usize>,
    #[serde(default)]
    byte_offset: usize,
    component_type: u32,
    #[serde(default)]
    normalized: bool,
    count: usize,
    #[serde(rename = "type")]
    accessor_type: String,
    sparse: Option<serde_json::Value>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct BufferViewSpec {
    buffer: usize,
    #[serde(default)]
    byte_offset: usize,
    byte_length: usize,
    byte_stride: Option<usize>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct BufferSpec {
    uri: Option<String>,
    byte_length: usize,
}
