//! Every format variant the benchmark compares, built from identical MVT bytes.

use anyhow::{Result, bail};
use mlt_core::encoder::{EncoderConfig, Network, Profile, Tessellation, WireVersion};
use mlt_core::mvt::mvt_to_tile_layers;

pub struct Variant {
    pub name: String,
    /// `None` for MVT.
    pub config: Option<EncoderConfig>,
}

pub fn variant(name: &str) -> Result<Variant> {
    let native = |version| {
        EncoderConfig::from_profile(Profile::Native(Network::Cellular4G))
            .with_wire_version(version)
            .with_tessellation(Tessellation::Off)
    };
    let web = |network| EncoderConfig::from_profile(Profile::Web(network)).with_wire_version(WireVersion::V02);
    let config = match name {
        "mvt" => None,
        // `mlt convert` defaults with triangles off: every codec, ranked by stored size.
        "v1" => Some(native(WireVersion::V01)),
        "v2" => Some(native(WireVersion::V02)),
        // The proposal: v1's codecs and predicted vertices, ranked by deflated size.
        "web" => Some(web(Network::Cellular4G)),
        // The uncompressed profile: the extensions, ranked by stored size.
        "offline" => Some(web(Network::Offline)),
        _ => bail!("unknown variant {name}"),
    };
    Ok(Variant { name: name.to_string(), config })
}

/// Encode an MVT tile as `v`, layer by layer, as `mlt convert` does.
pub fn encode(mvt: &[u8], v: &Variant) -> Result<Vec<u8>> {
    let Some(config) = v.config else {
        return Ok(mvt.to_vec());
    };
    let mut out = Vec::new();
    for layer in mvt_to_tile_layers(mvt)? {
        out.extend_from_slice(&layer.encode(config)?);
    }
    Ok(out)
}
