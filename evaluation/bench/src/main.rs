mod canon;
mod compress;
mod mvt;
mod variants;
mod verify;

use std::fmt::Write as _;
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::{Context as _, Result, bail};
use rayon::prelude::*;

use crate::variants::Variant;

pub struct Tile {
    pub id: String,
    pub mvt: Vec<u8>,
}

/// Every `*.mvt` / `*.pbf` file of `dir`, gunzipped if needed, sorted by name.
pub fn load_tiles(dir: &Path) -> Result<Vec<Tile>> {
    let mut tiles = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let path = entry?.path();
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if ext != "mvt" && ext != "pbf" {
            continue;
        }
        let mut data = std::fs::read(&path)?;
        if data.starts_with(&[0x1f, 0x8b]) {
            let mut out = Vec::new();
            flate2::read::GzDecoder::new(&data[..]).read_to_end(&mut out)?;
            data = out;
        }
        let id = path.file_stem().unwrap().to_string_lossy().into_owned();
        tiles.push(Tile { id, mvt: data });
    }
    tiles.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(tiles)
}

struct Args {
    rest: Vec<String>,
}

impl Args {
    fn flag(&mut self, name: &str) -> bool {
        if let Some(i) = self.rest.iter().position(|a| a == name) {
            self.rest.remove(i);
            true
        } else {
            false
        }
    }
    fn opt(&mut self, name: &str) -> Option<String> {
        let i = self.rest.iter().position(|a| a == name)?;
        self.rest.remove(i);
        Some(self.rest.remove(i))
    }
    fn req(&mut self, name: &str) -> Result<String> {
        self.opt(name).with_context(|| format!("missing {name}"))
    }
}

fn main() -> Result<()> {
    let mut all: Vec<String> = std::env::args().skip(1).collect();
    if all.is_empty() {
        bail!("usage: mltlab sizes|selftest ...");
    }
    let cmd = all.remove(0);
    let mut args = Args { rest: all };
    match cmd.as_str() {
        "sizes" => cmd_sizes(&mut args),
        "selftest" => cmd_selftest(&mut args),
        _ => bail!("unknown command {cmd}"),
    }
}

/// Inject known corruptions into correctly decoded tiles and require the strict verifier to catch each one.
fn cmd_selftest(args: &mut Args) -> Result<()> {
    use canon::{CLayer, CVal};
    let dir = PathBuf::from(args.req("--tiles")?);
    let tiles = load_tiles(&dir)?;
    let v = variants::variant(&args.opt("--variant").unwrap_or_else(|| "web".into()))?;
    type Mutation = (&'static str, fn(&mut Vec<CLayer>) -> bool);
    let mutations: [Mutation; 9] = [
        ("swap two features", |ls| {
            ls.iter_mut().find(|l| l.features.len() > 1 && l.features[0] != l.features[1]).is_some_and(|l| {
                l.features.swap(0, 1);
                true
            })
        }),
        ("move one vertex by 1", |ls| {
            ls.iter_mut().flat_map(|l| &mut l.features).flat_map(|f| &mut f.parts).find(|p| !p.is_empty()).is_some_and(|p| {
                p[0].0 += 1;
                true
            })
        }),
        ("rotate a ring start", |ls| {
            ls.iter_mut().flat_map(|l| &mut l.features).filter(|f| f.gtype == 3).flat_map(|f| &mut f.parts).find(|p| p.len() > 2).is_some_and(|p| {
                p.rotate_left(1);
                true
            })
        }),
        ("reverse a line", |ls| {
            ls.iter_mut().flat_map(|l| &mut l.features).filter(|f| f.gtype == 2).flat_map(|f| &mut f.parts).find(|p| p.len() > 1 && p.first() != p.last()).is_some_and(|p| {
                p.reverse();
                true
            })
        }),
        ("missing id becomes 0", |ls| {
            ls.iter_mut().flat_map(|l| &mut l.features).find(|f| f.id.is_none()).is_some_and(|f| {
                f.id = Some(0);
                true
            })
        }),
        ("int becomes float", |ls| {
            ls.iter_mut().flat_map(|l| &mut l.features).flat_map(|f| &mut f.props).find(|(_, v)| matches!(v, CVal::Int(_))).is_some_and(|(_, v)| {
                if let CVal::Int(i) = *v {
                    *v = CVal::Float((i as f64).to_bits());
                }
                true
            })
        }),
        ("drop an empty string", |ls| {
            ls.iter_mut().flat_map(|l| &mut l.features).find(|f| f.props.iter().any(|(_, v)| *v == CVal::Str(String::new()))).is_some_and(|f| {
                f.props.retain(|(_, v)| *v != CVal::Str(String::new()));
                true
            })
        }),
        ("drop a feature", |ls| {
            ls.iter_mut().find(|l| !l.features.is_empty()).is_some_and(|l| {
                l.features.pop();
                true
            })
        }),
        ("swap props between ID-less features", |ls| {
            ls.iter_mut().find(|l| l.features.len() > 1 && l.features[0].id.is_none() && l.features[1].id.is_none() && l.features[0].props != l.features[1].props).is_some_and(|l| {
                let p = std::mem::take(&mut l.features[0].props);
                l.features[0].props = std::mem::replace(&mut l.features[1].props, p);
                true
            })
        }),
    ];
    for (name, mutate) in mutations {
        let (mut applied, mut caught) = (0, 0);
        for tile in &tiles {
            let truth = mvt::parse_tile(&tile.mvt)?;
            let mut decoded = verify::decode_mlt(&variants::encode(&tile.mvt, &v)?)?;
            verify::strict(&truth, &decoded)?;
            if mutate(&mut decoded) {
                applied += 1;
                if verify::strict(&truth, &decoded).is_err() {
                    caught += 1;
                }
            }
        }
        eprintln!("{name}: applied to {applied} tiles, caught {caught}");
        if caught != applied {
            bail!("self-test failed for {name}");
        }
    }
    eprintln!("self-test passed");
    Ok(())
}

/// Encode every tile as every variant, verify it, and write per-tile raw and compressed sizes.
fn cmd_sizes(args: &mut Args) -> Result<()> {
    let dataset = args.req("--dataset")?;
    let dir = PathBuf::from(args.req("--tiles")?);
    let out = PathBuf::from(args.req("--out")?);
    let skip_slow = args.flag("--fast");
    let names = args.req("--variants")?;
    let variants: Vec<Variant> = names
        .split(',')
        .map(variants::variant)
        .collect::<Result<_>>()?;
    let tiles = load_tiles(&dir)?;
    eprintln!("{dataset}: {} tiles, {} variants", tiles.len(), variants.len());
    let failures = AtomicUsize::new(0);
    let skipped = std::sync::Mutex::new(Vec::<String>::new());
    let started = Instant::now();
    let rows: Vec<String> = tiles
        .par_iter()
        .flat_map_iter(|tile| {
            let truth = mvt::parse_tile(&tile.mvt)
                .with_context(|| format!("ground-truth parse of {}", tile.id));
            let mut rows = Vec::new();
            let truth = match truth {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("FAIL {dataset}/{}: {e:#}", tile.id);
                    failures.fetch_add(1, Ordering::Relaxed);
                    return rows;
                }
            };
            // MLT v2 stores a layer's extent as a power of two from 64 to 2^21. A tile with any
            // other extent cannot be written by any v2 variant, so it is left out for every
            // format, MVT and v1 included, and listed.
            if let Some(l) = truth.iter().find(|l| {
                !l.features.is_empty() && !(l.extent.is_power_of_two() && (64..=1 << 21).contains(&l.extent))
            }) {
                skipped.lock().unwrap().push(format!("{},{},{}", tile.id, l.name, l.extent));
                return rows;
            }
            for v in &variants {
                let t0 = Instant::now();
                let encoded = match variants::encode(&tile.mvt, v) {
                    Ok(e) => e,
                    Err(e) => {
                        eprintln!("FAIL {dataset}/{} {}: encode: {e:#}", tile.id, v.name);
                        failures.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                };
                let enc_ns = t0.elapsed().as_nanos();
                if v.config.is_some() {
                    let checked = verify::decode_mlt(&encoded).and_then(|decoded| verify::strict(&truth, &decoded));
                    if let Err(e) = checked {
                        eprintln!("FAIL {dataset}/{} {}: {e:#}", tile.id, v.name);
                        failures.fetch_add(1, Ordering::Relaxed);
                    }
                }
                let c = compress::sizes(&encoded, skip_slow);
                let mut row = String::new();
                write!(row, "{dataset},{},{},{}", tile.id, v.name, encoded.len()).unwrap();
                for s in c {
                    write!(row, ",{s}").unwrap();
                }
                write!(row, ",{enc_ns}").unwrap();
                rows.push(row);
            }
            rows
        })
        .collect();
    let mut csv = format!("dataset,tile,variant,raw,{},enc_ns\n", compress::CODECS.join(","));
    for r in rows {
        csv.push_str(&r);
        csv.push('\n');
    }
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&out, csv)?;
    let skipped = skipped.into_inner().unwrap();
    if !skipped.is_empty() {
        eprintln!("{dataset}: {} tiles left out, an extent v2 cannot store", skipped.len());
        std::fs::write(out.with_extension("skipped.csv"), format!("tile,layer,extent\n{}\n", skipped.join("\n")))?;
    }
    let duplicates = mvt::DUPLICATE_KEYS.load(Ordering::Relaxed);
    if duplicates > 0 {
        eprintln!("{dataset}: {duplicates} property keys repeated within a feature, read as the last value");
    }
    let failed = failures.load(Ordering::Relaxed);
    eprintln!(
        "{dataset}: done in {:.1}s, {failed} failures -> {}",
        started.elapsed().as_secs_f64(),
        out.display()
    );
    if failed > 0 {
        bail!("{failed} verification failures");
    }
    Ok(())
}
