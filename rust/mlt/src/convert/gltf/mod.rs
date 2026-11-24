//! `mlt convert <model.glb|model.gltf|dir|landmarks.toml> <dir>`: tile `glTF` 2.0 meshes into v2 `Meshes` layers.

mod geodesy;
mod reader;
mod tiler;

use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::ops::RangeInclusive;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;

use anyhow::{Context as _, Result as AnyResult, bail, ensure};
use clap::ValueEnum;
use martin_tile_utils::MAX_ZOOM;
use mlt_core::ZStep;
use mlt_core::geojson::PropertySchema;
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{Map, Value};
use walkdir::WalkDir;

use self::reader::Model;
use self::tiler::{
    COLOR, HEIGHT, Landmark, LayerSpec, LodModel, NAME, PART, Part, PartRules, Placement,
};
use super::{ConvertArgs, Reencoder};

/// Default z grid without `--z-step`: 1 cm.
const DEFAULT_Z_STEP: i8 = -2;
/// `Web Mercator` latitude ceiling in degrees (`atan(sinh(pi))`).
const MAX_MERCATOR_LAT: f64 = 85.051_128_779_806_6;

/// Which kind of `glTF` input `mlt convert` is reading.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Format {
    /// One `.gltf` / `.glb` file, or a directory of them.
    Model,
    /// A `TOML` landmark manifest referencing one or more `.gltf` / `.glb` models.
    Manifest,
}

/// Which vertical datum in a `glTF` model stands on the terrain at `z = 0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum Ground {
    /// The model's `y = 0` origin (or `CESIUM_RTC` centre), keeping subterranean geometry below zero.
    Origin,
    /// The lowest vertex of the model.
    Lowest,
}

/// `WGS84` `[lon_deg, lat_deg]` parsed from `--anchor lon,lat`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Anchor(pub [f64; 2]);

impl FromStr for Anchor {
    type Err = String;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        let (lon_s, lat_s) = raw
            .split_once(',')
            .ok_or_else(|| format!("expected `lon,lat` in degrees, got {raw:?}"))?;
        let lon: f64 = lon_s
            .trim()
            .parse()
            .map_err(|e| format!("invalid longitude {lon_s:?}: {e}"))?;
        let lat: f64 = lat_s
            .trim()
            .parse()
            .map_err(|e| format!("invalid latitude {lat_s:?}: {e}"))?;
        validate_anchor([lon, lat]).map_err(|e| e.to_string())?;
        Ok(Self([lon, lat]))
    }
}

fn validate_anchor([lon, lat]: [f64; 2]) -> AnyResult<()> {
    ensure!(
        lon.is_finite() && (-180.0..=180.0).contains(&lon),
        "anchor longitude {lon} must be within -180..=180 degrees"
    );
    ensure!(
        lat.is_finite() && (-MAX_MERCATOR_LAT..=MAX_MERCATOR_LAT).contains(&lat),
        "anchor latitude {lat} must be within -{MAX_MERCATOR_LAT:.5}..={MAX_MERCATOR_LAT:.5} degrees"
    );
    Ok(())
}

fn is_gltf_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("gltf") || e.eq_ignore_ascii_case("glb"))
}

fn is_toml_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("toml"))
}

/// Detects whether `path` is a `.gltf`/`.glb` file, a landmark `.toml` manifest, or a directory of `.gltf`/`.glb` files.
pub(super) fn input_format(path: &Path) -> AnyResult<Option<Format>> {
    if is_gltf_file(path) {
        return Ok(Some(Format::Model));
    }
    if is_toml_file(path) {
        return Ok(Some(Format::Manifest));
    }
    if !path.is_dir() {
        return Ok(None);
    }
    for entry in WalkDir::new(path).sort_by_file_name() {
        let entry = entry.with_context(|| format!("searching {}", path.display()))?;
        if !entry.file_type().is_file() {
            continue;
        }
        let file = entry.path();
        if is_gltf_file(file) {
            return Ok(Some(Format::Model));
        }
        if matches!(
            file.extension().and_then(|x| x.to_str()),
            Some("mlt" | "mvt" | "pbf")
        ) {
            return Ok(None);
        }
    }
    Ok(None)
}

pub(super) fn convert(args: &ConvertArgs, format: Format, reencoder: &Reencoder) -> AnyResult<()> {
    let input = &args.input;
    let output = &args.output;
    let z_exp = args.z_step.unwrap_or(DEFAULT_Z_STEP);
    let z_step = ZStep::new(z_exp)
        .with_context(|| format!("--z-step must be in -3..=4 (1 mm to 10 km), got {z_exp}"))?;

    let loaded = match format {
        Format::Model => load_model_files(args)?,
        Format::Manifest => load_manifest(args)?,
    };

    let spec = LayerSpec {
        name: loaded.layer_name,
        z_step,
        schema: property_schema(&loaded.landmarks)?,
        part_rules: loaded.part_rules,
    };
    let min_zoom = *loaded.zooms.start();
    let max_zoom = *loaded.zooms.end();
    let written = tiler::tile(&loaded.landmarks, &spec, loaded.zooms, output, reencoder)?;
    if written == 0 {
        eprintln!(
            "note: no tiles were written (the glTF input held no triangles at zooms {min_zoom}..={max_zoom})"
        );
    }
    eprintln!(
        "{} -> {}: wrote {written} tile(s) holding {} model(s) of {} file(s) at zooms {min_zoom}..={max_zoom}",
        input.display(),
        output.display(),
        loaded.landmarks.len(),
        loaded.file_count,
    );
    Ok(())
}

struct LoadedInput {
    layer_name: String,
    zooms: RangeInclusive<u8>,
    part_rules: PartRules,
    landmarks: Vec<Landmark>,
    file_count: usize,
}

fn load_model_files(args: &ConvertArgs) -> AnyResult<LoadedInput> {
    let (min_zoom, max_zoom) = args.tiled_zooms()?;
    let layer_name = args.tiled_layer_name()?;
    let files = collect_gltf_files(&args.input)?;
    ensure!(
        !files.is_empty(),
        "no .gltf or .glb files found in {}",
        args.input.display()
    );
    let models = read_unique_models(&files)?;
    let cli_anchor = args.anchor.map(|a| a.0);

    let mut landmarks = Vec::with_capacity(files.len());
    for (idx, path) in files.iter().enumerate() {
        let model = Arc::clone(&models[path]);
        let placement = resolve_placement(cli_anchor, model.rtc_center, path)?;
        let ground = args.ground.unwrap_or(match placement {
            Placement::Anchor(_) => Ground::Origin,
            Placement::Rtc(_) => Ground::Lowest,
        });
        let name = path.file_stem().and_then(|s| s.to_str()).map(str::to_owned);
        landmarks.push(Landmark {
            id: idx as u64,
            name,
            attributes: Map::new(),
            models: vec![LodModel {
                zooms: min_zoom..=max_zoom,
                placement,
                ground,
                model,
            }],
        });
    }

    Ok(LoadedInput {
        layer_name,
        zooms: min_zoom..=max_zoom,
        part_rules: PartRules::new(&[])?,
        landmarks,
        file_count: models.len(),
    })
}

fn collect_gltf_files(input: &Path) -> AnyResult<Vec<PathBuf>> {
    if input.is_file() {
        return Ok(vec![input.to_path_buf()]);
    }
    let mut out = Vec::new();
    for entry in WalkDir::new(input).sort_by_file_name() {
        let entry = entry.with_context(|| format!("searching {}", input.display()))?;
        if entry.file_type().is_file() && is_gltf_file(entry.path()) {
            out.push(entry.into_path());
        }
    }
    Ok(out)
}

fn load_manifest(args: &ConvertArgs) -> AnyResult<LoadedInput> {
    let raw = fs::read_to_string(&args.input)
        .with_context(|| format!("reading {}", args.input.display()))?;
    let manifest: ManifestSpec =
        toml::from_str(&raw).with_context(|| format!("parsing {}", args.input.display()))?;
    ensure!(
        !manifest.landmarks.is_empty(),
        "{} defines no [[landmarks]] entries",
        args.input.display()
    );
    if let (Some(min_z), Some(max_z)) = (args.min_zoom, args.max_zoom) {
        ensure!(
            min_z <= max_z,
            "--min-zoom ({min_z}) must be <= --max-zoom ({max_z})"
        );
    }
    if let Some(max_z) = args.max_zoom {
        ensure!(
            max_z <= MAX_ZOOM,
            "--max-zoom ({max_z}) must be <= {MAX_ZOOM}"
        );
    }

    let base_dir = args.input.parent().unwrap_or_else(|| Path::new("."));
    let layer_name = match (&args.layer, manifest.layer) {
        (Some(cli), _) => cli.clone(),
        (None, Some(from_toml)) => from_toml,
        (None, None) => args.tiled_layer_name()?,
    };

    let mut explicit_rules: Vec<(String, Part)> = Vec::new();
    for rule in &manifest.parts {
        let part = Part::parse(&rule.part)?;
        ensure!(
            rule.pattern.is_some() || !rule.nodes.is_empty(),
            "each [[parts]] entry in {} must set `match` or `nodes`",
            args.input.display()
        );
        if let Some(pat) = &rule.pattern {
            explicit_rules.push((pat.clone(), part));
        }
        for node_pat in &rule.nodes {
            explicit_rules.push((node_pat.clone(), part));
        }
    }
    let part_rules = PartRules::new(&explicit_rules)?;

    let mut paths_to_read: Vec<PathBuf> = Vec::new();
    for entry in &manifest.landmarks {
        match (&entry.gltf, entry.models.is_empty()) {
            (Some(p), true) => paths_to_read.push(base_dir.join(p)),
            (None, false) => {
                for m in &entry.models {
                    paths_to_read.push(base_dir.join(&m.gltf));
                }
            }
            (Some(_), false) => {
                bail!(
                    "landmark {:?} sets both `gltf` and `models`; pass one or the other",
                    entry.name.as_deref().unwrap_or("<unnamed>")
                );
            }
            (None, true) => {
                bail!(
                    "landmark {:?} must set either `gltf` or `models`",
                    entry.name.as_deref().unwrap_or("<unnamed>")
                );
            }
        }
    }
    let models_by_path = read_unique_models(&paths_to_read)?;

    let cli_anchor = args.anchor.map(|a| a.0);
    let mut used_ids = HashSet::new();
    let mut next_auto_id = 0u64;
    let (mut overall_min, mut overall_max) = (u8::MAX, 0u8);
    let mut landmarks = Vec::with_capacity(manifest.landmarks.len());

    for entry in manifest.landmarks {
        if let Some(anchor) = entry.anchor {
            validate_anchor(anchor)?;
        }
        let id = if let Some(explicit_id) = entry.id {
            ensure!(
                used_ids.insert(explicit_id),
                "duplicate landmark id {explicit_id} in {}",
                args.input.display()
            );
            explicit_id
        } else {
            while !used_ids.insert(next_auto_id) {
                next_auto_id += 1;
            }
            next_auto_id
        };

        let raw_entries: Vec<(PathBuf, u8, u8)> = if let Some(rel) = entry.gltf {
            let min_z = entry.min_zoom.or(args.min_zoom).unwrap_or(0);
            let max_z = entry.max_zoom.or(args.max_zoom).with_context(|| {
                format!(
                    "landmark {:?} uses `gltf` without `max-zoom`; set `max-zoom` on the landmark or pass `--max-zoom`",
                    entry.name.as_deref().unwrap_or("<unnamed>")
                )
            })?;
            vec![(base_dir.join(rel), min_z, max_z)]
        } else {
            ensure!(
                entry.min_zoom.is_none() && entry.max_zoom.is_none(),
                "landmark {:?} sets `models` alongside top-level `min-zoom`/`max-zoom`; put zoom ranges on each model entry",
                entry.name.as_deref().unwrap_or("<unnamed>")
            );
            entry
                .models
                .into_iter()
                .map(|m| (base_dir.join(m.gltf), m.min_zoom, m.max_zoom))
                .collect()
        };

        let mut covered_zooms = [false; (MAX_ZOOM as usize) + 1];
        let mut lod_models = Vec::with_capacity(raw_entries.len());
        for (path, min_z, max_z) in raw_entries {
            ensure!(
                min_z <= max_z,
                "landmark {:?} has min-zoom ({min_z}) > max-zoom ({max_z})",
                entry.name.as_deref().unwrap_or("<unnamed>")
            );
            ensure!(
                max_z <= MAX_ZOOM,
                "landmark {:?} has max-zoom ({max_z}) > {MAX_ZOOM}",
                entry.name.as_deref().unwrap_or("<unnamed>")
            );
            for z in min_z..=max_z {
                ensure!(
                    !std::mem::replace(&mut covered_zooms[z as usize], true),
                    "landmark {:?} lists overlapping models at zoom {z}",
                    entry.name.as_deref().unwrap_or("<unnamed>")
                );
            }
            let model = Arc::clone(&models_by_path[&path]);
            let placement =
                resolve_placement(entry.anchor.or(cli_anchor), model.rtc_center, &path)?;
            let ground =
                args.ground
                    .or(entry.ground)
                    .or(manifest.ground)
                    .unwrap_or(match placement {
                        Placement::Anchor(_) => Ground::Origin,
                        Placement::Rtc(_) => Ground::Lowest,
                    });
            overall_min = overall_min.min(min_z);
            overall_max = overall_max.max(max_z);
            lod_models.push(LodModel {
                zooms: min_z..=max_z,
                placement,
                ground,
                model,
            });
        }

        landmarks.push(Landmark {
            id,
            name: entry.name,
            attributes: entry.properties,
            models: lod_models,
        });
    }

    let min_zoom = args.min_zoom.unwrap_or(overall_min);
    let max_zoom = args.max_zoom.unwrap_or(overall_max);
    ensure!(
        min_zoom <= max_zoom,
        "--min-zoom ({min_zoom}) must be <= --max-zoom ({max_zoom})"
    );

    Ok(LoadedInput {
        layer_name,
        zooms: min_zoom..=max_zoom,
        part_rules,
        landmarks,
        file_count: models_by_path.len(),
    })
}

fn resolve_placement(
    anchor: Option<[f64; 2]>,
    rtc_center: Option<[f64; 3]>,
    path: &Path,
) -> AnyResult<Placement> {
    if let Some(anchor) = anchor {
        Ok(Placement::Anchor(anchor))
    } else if let Some(rtc) = rtc_center {
        Ok(Placement::Rtc(rtc))
    } else {
        bail!(
            "glTF model {} has no CESIUM_RTC extension; pass --anchor <lon,lat> or set `anchor = [lon, lat]` in a landmark .toml manifest",
            path.display()
        )
    }
}

fn read_unique_models(paths: &[PathBuf]) -> AnyResult<BTreeMap<PathBuf, Arc<Model>>> {
    let unique: Vec<PathBuf> = paths
        .iter()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    let parsed: Vec<(PathBuf, Arc<Model>)> = unique
        .into_par_iter()
        .map(|path| {
            let model = reader::read(&path)?;
            Ok((path, Arc::new(model)))
        })
        .collect::<AnyResult<_>>()?;
    Ok(parsed.into_iter().collect())
}

fn property_schema(landmarks: &[Landmark]) -> AnyResult<PropertySchema> {
    for landmark in landmarks {
        if let Some(reserved) = [NAME, HEIGHT, PART, COLOR]
            .into_iter()
            .find(|&col| landmark.attributes.contains_key(col))
        {
            bail!(
                "landmark {:?} defines a property named {reserved:?}, which convert writes a column of its own under",
                landmark.name.as_deref().unwrap_or("<unnamed>")
            );
        }
    }
    Ok(PropertySchema::infer(
        landmarks.iter().map(|l| l.attributes.iter()),
    )?)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
struct ManifestSpec {
    layer: Option<String>,
    #[expect(
        dead_code,
        reason = "accepted for compatibility with landmark manifests carrying dataset attribution"
    )]
    attribution: Option<String>,
    ground: Option<Ground>,
    #[serde(default)]
    parts: Vec<PartRuleSpec>,
    #[serde(default)]
    landmarks: Vec<LandmarkEntrySpec>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PartRuleSpec {
    #[serde(rename = "match")]
    pattern: Option<String>,
    #[serde(default)]
    nodes: Vec<String>,
    part: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
struct LandmarkEntrySpec {
    id: Option<u64>,
    name: Option<String>,
    anchor: Option<[f64; 2]>,
    ground: Option<Ground>,
    gltf: Option<PathBuf>,
    min_zoom: Option<u8>,
    max_zoom: Option<u8>,
    #[serde(default)]
    models: Vec<ModelEntrySpec>,
    #[serde(default)]
    properties: Map<String, Value>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
struct ModelEntrySpec {
    gltf: PathBuf,
    min_zoom: u8,
    max_zoom: u8,
}
