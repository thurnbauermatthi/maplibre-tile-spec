//! Per-tile general-purpose compression, as a tile is served over HTTP.

use std::io::Write as _;

use flate2::Compression;
use flate2::write::GzEncoder;

pub const CODECS: [&str; 5] = ["gz6", "br11", "br5", "zs19", "zs3"];

pub fn gzip(data: &[u8], level: u32) -> Vec<u8> {
    let mut e = GzEncoder::new(Vec::with_capacity(data.len() / 2 + 64), Compression::new(level));
    e.write_all(data).expect("in-memory gzip");
    e.finish().expect("in-memory gzip")
}

pub fn brotli(data: &[u8], quality: i32) -> Vec<u8> {
    let params = brotli::enc::BrotliEncoderParams {
        quality,
        lgwin: 22,
        size_hint: data.len(),
        ..Default::default()
    };
    let mut out = Vec::with_capacity(data.len() / 2 + 64);
    brotli::BrotliCompress(&mut &data[..], &mut out, &params).expect("in-memory brotli");
    out
}

pub fn zstd(data: &[u8], level: i32) -> Vec<u8> {
    zstd::bulk::compress(data, level).expect("in-memory zstd")
}

/// Compressed sizes in [`CODECS`] order. `skip_slow` leaves brotli 11 and zstd 19 at 0.
pub fn sizes(data: &[u8], skip_slow: bool) -> [usize; 5] {
    [
        gzip(data, 6).len(),
        if skip_slow { 0 } else { brotli(data, 11).len() },
        brotli(data, 5).len(),
        if skip_slow { 0 } else { zstd(data, 19).len() },
        zstd(data, 3).len(),
    ]
}
