//! Round-trip and differential tests for the experimental v2 (tag `0x02`) wire format.

use insta::{assert_snapshot, with_settings};
use mlt_core::dump::{DumpTree, RenderOpts, annotate_tile, render};
use mlt_core::encoder::{EncoderConfig, Tessellation, WireVersion};
use mlt_core::encoder::{Network, Profile};
use mlt_core::geo_types::{
    Coord, Geometry, LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon,
};
use mlt_core::{Decoder, Layer, Parser, PropValue, TileFeature, TileLayer};
use rstest::rstest;

/// The full annotation of `bytes`, which the walker must produce without bailing.
fn annotate(bytes: &[u8]) -> DumpTree {
    let (tree, err) = annotate_tile(bytes);
    assert!(err.is_none(), "annotate_tile: {err:?}");
    tree
}

fn cfg_v1() -> EncoderConfig {
    // sorting because sort trials could otherwise pick different winners per format.
    EncoderConfig::from_profile(Profile::Native(Network::Cellular4G))
}

fn cfg_v2() -> EncoderConfig {
    cfg_v1().with_wire_version(WireVersion::V02)
}

#[expect(clippy::wildcard_enum_match_arm, reason = "Layer is non_exhaustive")]
fn decode(bytes: &[u8]) -> (u8, TileLayer) {
    let mut parser = Parser::default();
    let layers = parser.parse_layers(bytes).expect("parse");
    assert_eq!(layers.len(), 1, "expected a single layer");
    let tag = match &layers[0] {
        Layer::Tag01(_) => 1,
        Layer::Tag02(_) => 2,
        _ => panic!("unexpected layer kind"),
    };
    let mut dec = Decoder::default();
    let tile = layers
        .into_iter()
        .next()
        .unwrap()
        .into_tile(&mut dec)
        .expect("into_tile")
        .expect("a known layer tag");
    (tag, tile)
}

fn assert_differential(layer: &TileLayer) -> (usize, usize) {
    assert_differential_with(layer, cfg_v1())
}

/// As [`assert_differential`], with `cfg` deciding everything but the wire version.
fn assert_differential_with(layer: &TileLayer, cfg: EncoderConfig) -> (usize, usize) {
    let v1_bytes = layer.encode(cfg).expect("v1 encode");
    let v2_bytes = layer
        .clone()
        .encode(cfg.with_wire_version(WireVersion::V02))
        .expect("v2 encode");
    let (tag1, tile1) = decode(&v1_bytes);
    let (tag2, tile2) = decode(&v2_bytes);
    assert_eq!(tag1, 1);
    assert_eq!(tag2, 2);
    assert_eq!(tile1, tile2, "v1 and v2 decoded layers must be identical");
    assert_dump_covers(&v1_bytes);
    assert_dump_covers(&v2_bytes);
    (v1_bytes.len(), v2_bytes.len())
}

/// The annotated dump of `bytes`, rendered as the `mlt dump` CLI would show it.
fn dump_text(bytes: &[u8]) -> String {
    let tree = annotate(bytes);
    let mut out = Vec::new();
    render(&tree, bytes, &RenderOpts::default(), &mut out).expect("render");
    String::from_utf8(out).expect("dump is utf8")
}

/// The value of every `label` field in the tile's dump, in wire order.
fn dump_fields(bytes: &[u8], label: &str) -> Vec<String> {
    dump_text(bytes)
        .lines()
        .filter_map(|line| line.split_once(label))
        .map(|(_, value)| value.trim().to_string())
        .collect()
}

fn assert_dump_covers(bytes: &[u8]) {
    let tree = annotate(bytes);
    let mut leaves: Vec<(usize, usize)> = tree
        .regions
        .iter()
        .filter(|r| !r.container)
        .map(|r| (r.offset, r.len))
        .collect();
    leaves.sort_unstable();

    let mut cursor = 0;
    for (offset, len) in &leaves {
        assert_eq!(*offset, cursor, "gap/overlap at offset {offset}");
        cursor += len;
    }
    assert_eq!(
        cursor,
        bytes.len(),
        "dump covers {cursor} of {} bytes",
        bytes.len()
    );

    render(&tree, bytes, &RenderOpts::default(), &mut Vec::new()).expect("render");
}

#[expect(clippy::needless_pass_by_value, reason = "test ergonomics")]
fn layer(
    geoms: Vec<Geometry<i32>>,
    ids: Option<Vec<Option<u64>>>,
    props: &[(&str, Vec<PropValue>)],
) -> TileLayer {
    let mut builder = TileLayer::builder("test_layer", 4096).unwrap();
    let keys: Vec<_> = props
        .iter()
        .map(|(name, values)| builder.add_property(*name, values[0].kind()).unwrap())
        .collect();
    for (i, geom) in geoms.into_iter().enumerate() {
        let mut feature = builder.feature(geom);
        if let Some(ids) = &ids {
            feature.id(ids[i]);
        }
        for (key, (_, values)) in keys.iter().zip(props) {
            feature.property(*key, values[i].clone()).unwrap();
        }
        feature.finish().unwrap();
    }
    builder.finish()
}

fn pt(x: i32, y: i32) -> Geometry<i32> {
    Geometry::Point(Point::new(x, y))
}

/// One point per character of a presence mask, so masks alone size a layer.
fn points(mask: &str) -> Vec<Geometry<i32>> {
    (0..mask.len())
        .map(|i| pt(i32::try_from(i).unwrap(), 0))
        .collect()
}

/// An optional `u32` column holding a value at every `'1'` of `mask`.
fn opt_col(mask: &str) -> Vec<PropValue> {
    mask.bytes()
        .enumerate()
        .map(|(i, b)| PropValue::U32((b == b'1').then(|| u32::try_from(i).unwrap())))
        .collect()
}

fn coords(pts: &[(i32, i32)]) -> Vec<Coord<i32>> {
    pts.iter().map(|&(x, y)| Coord { x, y }).collect()
}

fn line(pts: &[(i32, i32)]) -> Geometry<i32> {
    Geometry::LineString(LineString::new(coords(pts)))
}

fn ring(pts: &[(i32, i32)]) -> LineString<i32> {
    let mut ls = LineString::new(coords(pts));
    ls.close();
    ls
}

#[rstest]
#[case::points(vec![pt(0, 0), pt(10, 20), pt(-5, 4000)])]
#[case::single_point(vec![pt(7, 9)])]
#[case::multipoints(vec![
    Geometry::MultiPoint(MultiPoint(vec![Point::new(1, 2), Point::new(3, 4)])),
    Geometry::MultiPoint(MultiPoint(vec![Point::new(5, 6)])),
    Geometry::MultiPoint(MultiPoint(vec![
        Point::new(7, 8),
        Point::new(9, 10),
        Point::new(11, 12),
    ])),
])]
#[case::linestrings(vec![
    line(&[(0, 0), (10, 10), (20, 5)]),
    line(&[(100, 100), (150, 200)]),
])]
#[case::multilinestrings(vec![
    Geometry::MultiLineString(MultiLineString(vec![
        LineString::new(coords(&[(0, 0), (5, 5)])),
        LineString::new(coords(&[(10, 10), (20, 20), (30, 15)])),
    ])),
    Geometry::MultiLineString(MultiLineString(vec![LineString::new(coords(&[
        (50, 50),
        (60, 40),
    ]))])),
])]
#[case::polygons_with_hole(vec![
    Geometry::Polygon(Polygon::new(
        ring(&[(0, 0), (100, 0), (100, 100), (0, 100)]),
        vec![ring(&[(20, 20), (40, 20), (40, 40), (20, 40)])],
    )),
    Geometry::Polygon(Polygon::new(
        ring(&[(200, 200), (300, 200), (250, 300)]),
        vec![],
    )),
])]
#[case::multipolygons(vec![
    Geometry::MultiPolygon(MultiPolygon(vec![
        Polygon::new(ring(&[(0, 0), (10, 0), (10, 10), (0, 10)]), vec![]),
        Polygon::new(ring(&[(20, 20), (30, 20), (30, 30)]), vec![]),
    ])),
    Geometry::MultiPolygon(MultiPolygon(vec![Polygon::new(
        ring(&[(50, 50), (60, 50), (60, 60), (50, 60)]),
        vec![ring(&[(52, 52), (57, 52), (57, 57)])],
    )])),
])]
#[case::mixed_points_and_lines(vec![pt(5, 5), line(&[(0, 0), (10, 10), (20, 0)]), pt(30, 30)])]
#[case::mixed_points_and_polygons(vec![
    pt(5, 5),
    Geometry::Polygon(Polygon::new(
        ring(&[(0, 0), (100, 0), (100, 100), (0, 100)]),
        vec![],
    )),
])]
#[case::mixed_polygon_and_multipolygon(vec![
    Geometry::Polygon(Polygon::new(
        ring(&[(0, 0), (10, 0), (10, 10), (0, 10)]),
        vec![],
    )),
    Geometry::MultiPolygon(MultiPolygon(vec![
        Polygon::new(ring(&[(20, 20), (30, 20), (30, 30)]), vec![]),
        Polygon::new(ring(&[(40, 40), (50, 40), (50, 50)]), vec![]),
    ])),
])]
fn geometry_types(#[case] geoms: Vec<Geometry<i32>>) {
    assert_differential(&layer(geoms, None, &[]));
}

#[rstest]
#[case::u32(vec![pt(0, 0), pt(1, 1), pt(2, 2)], vec![Some(1), Some(2), Some(3)])]
#[case::sequential_delta_friendly(
    (0..50).map(|i| pt(i, i * 2)).collect(),
    (0..50_u64).map(|i| Some(1_000_000 + i)).collect(),
)]
#[case::u64_large(
    vec![pt(0, 0), pt(1, 1)],
    vec![Some(u64::from(u32::MAX) + 10), Some(u64::MAX - 5)],
)]
#[case::optional_with_nulls(
    vec![pt(0, 0), pt(1, 1), pt(2, 2), pt(3, 3)],
    vec![None, Some(7), None, Some(9)],
)]
fn ids(#[case] geoms: Vec<Geometry<i32>>, #[case] feature_ids: Vec<Option<u64>>) {
    assert_differential(&layer(geoms, Some(feature_ids), &[]));
}

#[test]
fn all_scalar_types_non_optional() {
    let geoms = vec![pt(0, 0), pt(1, 1), pt(2, 2)];
    let props = [
        (
            "b",
            vec![true, false, true]
                .into_iter()
                .map(|v| PropValue::Bool(Some(v)))
                .collect(),
        ),
        (
            "i32",
            vec![-100_000_i32, 0, 100_000]
                .into_iter()
                .map(|v| PropValue::I32(Some(v)))
                .collect(),
        ),
        (
            "u32",
            vec![0_u32, 70_000, u32::MAX]
                .into_iter()
                .map(|v| PropValue::U32(Some(v)))
                .collect(),
        ),
        (
            "i64",
            vec![i64::MIN, 0, i64::MAX]
                .into_iter()
                .map(|v| PropValue::I64(Some(v)))
                .collect(),
        ),
        (
            "u64",
            vec![0_u64, 1, u64::MAX]
                .into_iter()
                .map(|v| PropValue::U64(Some(v)))
                .collect(),
        ),
        (
            "f32",
            vec![-1.5_f32, 0.0, 3.25]
                .into_iter()
                .map(|v| PropValue::F32(Some(v)))
                .collect(),
        ),
        (
            "f64",
            vec![-2.5_f64, 0.0, 1e100]
                .into_iter()
                .map(|v| PropValue::F64(Some(v)))
                .collect(),
        ),
    ];
    assert_differential(&layer(geoms, None, &props));
}

#[test]
fn eight_bit_ints_widen_in_v1_only() {
    let geoms = vec![pt(0, 0), pt(1, 1)];
    let props = [
        (
            "i8",
            vec![PropValue::I8(Some(i8::MIN)), PropValue::I8(None)],
        ),
        (
            "u8",
            vec![PropValue::U8(None), PropValue::U8(Some(u8::MAX))],
        ),
    ];
    let src = layer(geoms, None, &props);
    let props_of = |cfg: EncoderConfig| -> Vec<Vec<PropValue>> {
        let (_, tile) = decode(&src.encode(cfg).expect("encode"));
        tile.features()
            .iter()
            .map(|f| f.properties().to_vec())
            .collect()
    };
    assert_eq!(
        props_of(cfg_v1()),
        [
            [PropValue::I32(Some(-128)), PropValue::U32(None)],
            [PropValue::I32(None), PropValue::U32(Some(255))],
        ]
    );
    assert_eq!(
        props_of(cfg_v2()),
        [
            [PropValue::I8(Some(i8::MIN)), PropValue::U8(None)],
            [PropValue::I8(None), PropValue::U8(Some(u8::MAX))],
        ]
    );
}

#[test]
fn all_scalar_types_optional_with_nulls() {
    let geoms = vec![pt(0, 0), pt(1, 1), pt(2, 2), pt(3, 3)];
    let props = [
        (
            "b",
            vec![
                PropValue::Bool(None),
                PropValue::Bool(Some(true)),
                PropValue::Bool(None),
                PropValue::Bool(Some(false)),
            ],
        ),
        (
            "i32",
            vec![
                PropValue::I32(None),
                PropValue::I32(Some(-1)),
                PropValue::I32(Some(1)),
                PropValue::I32(None),
            ],
        ),
        (
            "u32",
            vec![
                PropValue::U32(Some(1)),
                PropValue::U32(Some(2)),
                PropValue::U32(None),
                PropValue::U32(Some(3)),
            ],
        ),
        (
            "i64",
            vec![
                PropValue::I64(None),
                PropValue::I64(Some(i64::MIN)),
                PropValue::I64(None),
                PropValue::I64(Some(i64::MAX)),
            ],
        ),
        (
            "u64",
            vec![
                PropValue::U64(Some(u64::MAX)),
                PropValue::U64(None),
                PropValue::U64(Some(0)),
                PropValue::U64(None),
            ],
        ),
        (
            "f32",
            vec![
                PropValue::F32(None),
                PropValue::F32(Some(1.5)),
                PropValue::F32(Some(-1.5)),
                PropValue::F32(None),
            ],
        ),
        (
            "f64",
            vec![
                PropValue::F64(Some(2.5)),
                PropValue::F64(None),
                PropValue::F64(None),
                PropValue::F64(Some(-2.5)),
            ],
        ),
    ];
    assert_differential(&layer(geoms, None, &props));
}

#[test]
fn rle_friendly_constant_column() {
    let n = 100;
    let geoms: Vec<_> = (0..n).map(|i| pt(i, i)).collect();
    let props = [
        (
            "const",
            (0..n).map(|_| PropValue::I32(Some(42))).collect::<Vec<_>>(),
        ),
        (
            "runs",
            (0..n)
                .map(|i| PropValue::U32(Some(u32::from(i > 50))))
                .collect(),
        ),
    ];
    assert_differential(&layer(geoms, None, &props));
}

#[test]
fn bool_column_is_packed_bitfield() {
    let n = 256_i32;
    let geoms: Vec<_> = (0..n).map(|i| pt(i, 0)).collect();
    let values: Vec<PropValue> = (0..n).map(|i| PropValue::Bool(Some(i % 2 == 0))).collect();
    let l = layer(geoms, None, &[("flag", values)]);
    let (_v1, v2) = assert_differential(&l);
    // Ceiling still fails if the 32 B bitfield regresses to a byte per bool.
    assert!(v2 < 900, "v2 tile unexpectedly large: {v2} B");
}

#[rstest]
fn presence_bitfield_byte_boundaries(#[values(1_u32, 7, 8, 9, 17)] n: u32) {
    #[expect(clippy::cast_possible_wrap, reason = "tiny test values")]
    let geoms: Vec<_> = (0..n).map(|i| pt(i as i32, 0)).collect();
    let values: Vec<PropValue> = (0..n)
        .map(|i| PropValue::U32((i % 2 == 0).then_some(i)))
        .collect();
    assert_differential(&layer(geoms, None, &[("alt", values)]));
}

#[test]
fn multiple_layers() {
    let a = layer(vec![pt(0, 0)], None, &[]);
    let b = layer(vec![pt(5, 5), pt(6, 6)], Some(vec![Some(1), Some(2)]), &[]);
    let mut tile = a.encode(cfg_v2()).unwrap();
    tile.extend_from_slice(&b.encode(cfg_v2()).unwrap());

    let mut parser = Parser::default();
    let layers = parser.parse_layers(&tile).expect("parse");
    assert_eq!(layers.len(), 2);
    let mut dec = Decoder::default();
    for l in layers {
        assert!(matches!(l, Layer::Tag02(_)));
        l.into_tile(&mut dec).unwrap().expect("a known layer tag");
    }
}

#[test]
fn native_preset_with_sort_trials() {
    let geoms: Vec<_> = (0..30).map(|i| pt(i * 13 % 100, i * 7 % 100)).collect();
    let ids = Some((0..30_u64).map(Some).collect());
    let l = layer(geoms, ids, &[]);

    let bytes = l
        .clone()
        .encode(
            EncoderConfig::from_profile(Profile::Native(Network::Cellular4G))
                .with_wire_version(WireVersion::V02),
        )
        .unwrap();
    let (tag, tile) = decode(&bytes);
    assert_eq!(tag, 2);
    assert_eq!(tile.feature_count(), l.feature_count());
    // Feature order may differ (sorting); compare as multisets by id.
    let mut expected: Vec<_> = l.features().iter().map(TileFeature::id).collect();
    let mut actual: Vec<_> = tile.features().iter().map(TileFeature::id).collect();
    expected.sort_unstable();
    actual.sort_unstable();
    assert_eq!(expected, actual);
}

#[test]
fn a_bitpacking_friendly_column_takes_fastpfor() {
    let n = 2000_u32;
    let geoms: Vec<_> = (0..n).map(|i| pt(i32::try_from(i).unwrap(), 0)).collect();
    let values: Vec<PropValue> = (0..n)
        .map(|i| PropValue::U32(Some(i.wrapping_mul(2_654_435_761) % 4096)))
        .collect();
    let l = layer(geoms, None, &[("scattered", values)]);

    let v2 = l.encode(cfg_v2()).expect("v2 encode");
    assert!(
        dump_text(&v2).contains("physical = FastPFor"),
        "v2 should code the scattered column with FastPFor"
    );
    assert_differential(&l);

    let without = l
        .encode(cfg_v2().with_fastpfor(false))
        .expect("v2 encode without fastpfor");
    assert!(
        v2.len() < without.len(),
        "FastPFor ({} B) should beat the varint fallback ({} B)",
        v2.len(),
        without.len()
    );
}

#[rstest]
fn a_fastpfor_stream_round_trips_across_block_boundaries(
    #[values(256, 257, 511, 512, 513, 767, 768, 769)] n: u32,
) {
    let geoms: Vec<_> = (0..n).map(|i| pt(i32::try_from(i).unwrap(), 0)).collect();
    let values: Vec<PropValue> = (0..n)
        .map(|i| PropValue::U32(Some(i.wrapping_mul(2_654_435_761) % 4096)))
        .collect();
    let l = layer(geoms, None, &[("scattered", values)]);
    let v2 = l.encode(cfg_v2()).expect("v2 encode");
    assert!(
        dump_text(&v2).contains("physical = FastPFor"),
        "{n} scattered values should be coded with FastPFor"
    );
    assert_differential(&l);
}

#[test]
fn v2_is_smaller_for_typical_layer() {
    let n = 100_u16;
    let geoms: Vec<_> = (0..n)
        .map(|i| pt(i32::from(i) * 3, i32::from(i) * 5))
        .collect();
    let ids = Some((0..u64::from(n)).map(|i| Some(1000 + i)).collect());
    // Alternating presence is RLE-hostile, favoring v2's raw bitfields.
    let props = [
        (
            "opt_a",
            (0..n)
                .map(|i| PropValue::U32((i % 2 == 0).then(|| u32::from(i))))
                .collect::<Vec<_>>(),
        ),
        (
            "opt_b",
            (0..n)
                .map(|i| PropValue::I32((i % 2 == 1).then(|| -i32::from(i))))
                .collect(),
        ),
        (
            "clazz",
            (0..n)
                .map(|i| PropValue::U32(Some(u32::from(i % 7 == 0))))
                .collect(),
        ),
        (
            "height",
            (0..n)
                .map(|i| PropValue::F32(Some(f32::from(i) * 0.5)))
                .collect(),
        ),
    ];
    let l = layer(geoms, ids, &props);
    let (v1_len, v2_len) = assert_differential(&l);
    assert!(
        v2_len < v1_len,
        "v2 ({v2_len} B) should be smaller than v1 ({v1_len} B)"
    );
}

#[test]
fn every_optional_column_carries_its_own_presence_bitfield() {
    const SAME: &str = "100100100100100100";
    let ids = SAME
        .bytes()
        .enumerate()
        .map(|(i, b)| (b == b'1').then(|| i as u64 + 1))
        .collect();
    let props = [("a", opt_col(SAME)), ("b", opt_col(SAME))];
    let l = layer(points(SAME), Some(ids), &props);
    assert_differential(&l);

    let dump = dump_text(&l.encode(cfg_v2()).unwrap());
    assert_eq!(dump.matches("presence = Inline").count(), 3, "{dump}");
    assert_eq!(dump.matches("[Present ").count(), 3, "{dump}");
}

mod geometry_layouts {
    use super::*;

    fn cfg_tessellated() -> EncoderConfig {
        cfg_v1().with_tessellation(Tessellation::Always)
    }

    fn square(x: i32, y: i32) -> Polygon<i32> {
        Polygon::new(
            ring(&[(x, y), (x + 10, y), (x + 10, y + 10), (x, y + 10)]),
            vec![],
        )
    }

    /// Points cycling through a handful of coordinates, so a vertex dictionary pays off.
    fn repeated_points(n: i32) -> Vec<Geometry<i32>> {
        (0..n).map(|i| pt((i % 7) * 10, (i % 5) * 10)).collect()
    }

    #[test]
    fn repeated_vertices_pick_a_dictionary_layout() {
        let l = layer(repeated_points(200), None, &[]);
        let dump = dump_text(&l.encode(cfg_v2()).unwrap());
        assert!(dump.contains("geometry layout = PointsDict"), "{dump}");
        assert!(dump.contains("vertex_offsets"), "{dump}");
        assert_differential(&l);
    }

    #[test]
    fn distinct_vertices_stay_plain() {
        let l = layer((0..64).map(|i| pt(i * 7, i * 13)).collect(), None, &[]);
        let dump = dump_text(&l.encode(cfg_v2()).unwrap());
        assert!(dump.contains("geometry layout = Points\n"), "{dump}");
        assert_differential(&l);
    }

    #[rstest]
    #[case::polygons(vec![
        Geometry::Polygon(square(0, 0)),
        Geometry::Polygon(Polygon::new(
            ring(&[(0, 0), (100, 0), (100, 100), (0, 100)]),
            vec![ring(&[(20, 20), (40, 20), (40, 40), (20, 40)])],
        )),
    ])]
    #[case::mixed_points_and_polygons(vec![pt(5, 5), Geometry::Polygon(square(20, 20))])]
    #[case::mixed_lines_and_polygons(vec![
        line(&[(0, 0), (10, 10), (20, 0)]),
        Geometry::Polygon(square(40, 40)),
    ])]
    #[case::multipolygons(vec![
        Geometry::MultiPolygon(MultiPolygon(vec![square(0, 0), square(20, 20)])),
        Geometry::Polygon(square(50, 50)),
    ])]
    fn tessellated_geometry(#[case] geoms: Vec<Geometry<i32>>) {
        let l = layer(geoms, None, &[]);
        let dump = dump_text(
            &l.clone()
                .encode(cfg_tessellated().with_wire_version(WireVersion::V02))
                .unwrap(),
        );
        assert!(
            dump.contains("geometry layout = TessPolygonsWithOutlines"),
            "{dump}"
        );
        assert_differential_with(&l, cfg_tessellated());
    }

    /// The meaning of every bit field of the layer's two header bytes, in wire order.
    fn header_bits(bytes: &[u8]) -> Vec<String> {
        annotate(bytes)
            .regions
            .iter()
            .filter(|r| r.label == "header" || r.label == "layout")
            .flat_map(|r| r.bits.iter().map(|b| b.meaning().to_string()))
            .collect()
    }

    /// The streams of the layer's geometry section, in wire order.
    fn geometry_streams(bytes: &[u8]) -> Vec<String> {
        let tree = annotate(bytes);
        let geometry = tree
            .regions
            .iter()
            .find(|r| r.label == "geometry")
            .expect("a geometry section");
        tree.regions
            .iter()
            .skip_while(|r| r.label != "geometry")
            .skip(1)
            .take_while(|r| r.depth > geometry.depth)
            .filter(|r| r.depth == geometry.depth + 1)
            .map(|r| r.label.clone())
            .collect()
    }

    #[rstest]
    #[case::points(vec![pt(1, 2), pt(3, 4)], "Point")]
    #[case::lines(vec![line(&[(0, 0), (1, 1)]), line(&[(2, 2), (3, 3)])], "LineString")]
    #[case::polygons(vec![Geometry::Polygon(square(0, 0))], "Polygon")]
    #[case::multi_points(
        vec![Geometry::MultiPoint(MultiPoint(vec![Point::new(1, 2), Point::new(3, 4)]))],
        "MultiPoint"
    )]
    #[case::multi_polygons(
        vec![Geometry::MultiPolygon(MultiPolygon(vec![square(0, 0), square(20, 20)]))],
        "MultiPolygon"
    )]
    fn one_geometry_type_moves_into_the_header_byte(
        #[case] geoms: Vec<Geometry<i32>>,
        #[case] geometry_type: &str,
    ) {
        let l = layer(geoms, None, &[]);
        let bytes = l.encode(cfg_v2()).unwrap();
        with_settings!({snapshot_suffix => geometry_type}, {
            assert_snapshot!(header_bits(&bytes).join("\n"));
        });
        assert!(!geometry_streams(&bytes).contains(&"types".to_string()));
        assert_differential(&l);
    }

    #[test]
    fn mixed_geometry_types_keep_their_stream() {
        let l = layer(vec![pt(5, 5), line(&[(0, 0), (10, 10)])], None, &[]);
        let bytes = l.encode(cfg_v2()).unwrap();
        assert_snapshot!(header_bits(&bytes).join("\n"), @"
        no m-value section
        a types stream leads the geometry section
        extent 2^(n+6) = 4096
        reserved
        geometry layout = Lines
        ");
        assert_snapshot!(geometry_streams(&bytes).join("\n"), @"
        types
        part_lengths
        vertices
        ");
        assert_differential(&l);
    }

    #[test]
    fn a_layer_without_polygons_is_not_tessellated() {
        let l = layer(vec![pt(5, 5), line(&[(0, 0), (10, 10)])], None, &[]);
        let dump = dump_text(
            &l.clone()
                .encode(cfg_tessellated().with_wire_version(WireVersion::V02))
                .unwrap(),
        );
        assert!(dump.contains("geometry layout = Lines"), "{dump}");
        assert_differential_with(&l, cfg_tessellated());
    }

    fn cfg_triangles_only() -> EncoderConfig {
        cfg_tessellated()
            .with_wire_version(WireVersion::V02)
            .with_triangles_only(true)
    }

    fn exteriors(tile: &TileLayer) -> Vec<Vec<Vec<(i32, i32)>>> {
        tile.features()
            .iter()
            .map(|f| {
                let Geometry::MultiPolygon(mp) = f.geometry() else {
                    panic!("expected a MultiPolygon, got {:?}", f.geometry());
                };
                mp.iter()
                    .map(|p| p.exterior().coords().map(|c| (c.x, c.y)).collect())
                    .collect()
            })
            .collect()
    }

    #[test]
    fn a_polygon_layer_stores_only_its_triangles() {
        let l = layer(
            vec![
                Geometry::Polygon(square(0, 0)),
                Geometry::MultiPolygon(MultiPolygon(vec![square(20, 20), square(40, 40)])),
            ],
            None,
            &[],
        );
        let bytes = l.encode(cfg_triangles_only()).unwrap();
        assert_snapshot!(header_bits(&bytes).join("\n"), @"
        no m-value section
        a types stream leads the geometry section
        extent 2^(n+6) = 4096
        reserved
        geometry layout = TessPolygons
        ");
        assert_snapshot!(geometry_streams(&bytes).join("\n"), @"
        types
        tri_lengths
        tri_symbols
        tri_cycles
        vertices
        ");
        let (_, tile) = decode(&bytes);
        assert_snapshot!(format!("{:?}", exteriors(&tile)), @"[[[(0, 0), (10, 0), (10, 10), (0, 0)], [(0, 0), (10, 10), (0, 10), (0, 0)]], [[(20, 20), (30, 20), (30, 30), (20, 20)], [(20, 20), (30, 30), (20, 30), (20, 20)], [(40, 40), (50, 40), (50, 50), (40, 40)], [(40, 40), (50, 50), (40, 50), (40, 40)]]]");
    }

    #[test]
    fn a_polygon_with_no_triangles_decodes_as_an_empty_multipolygon() {
        let collinear = Polygon::new(ring(&[(0, 0), (10, 0), (20, 0)]), vec![]);
        let l = layer(
            vec![
                Geometry::Polygon(collinear),
                Geometry::Polygon(square(40, 40)),
            ],
            None,
            &[],
        );
        let (_, tile) = decode(&l.encode(cfg_triangles_only()).unwrap());
        assert_snapshot!(format!("{:?}", exteriors(&tile)), @"[[], [[(40, 40), (50, 40), (50, 50), (40, 40)], [(40, 40), (50, 50), (40, 50), (40, 40)]]]");
    }

    #[test]
    fn dropping_the_outlines_is_smaller() {
        let l = layer(
            vec![
                Geometry::Polygon(square(0, 0)),
                Geometry::MultiPolygon(MultiPolygon(vec![square(20, 20), square(40, 40)])),
            ],
            None,
            &[],
        );
        let with_outlines = l
            .clone()
            .encode(cfg_tessellated().with_wire_version(WireVersion::V02))
            .unwrap();
        let triangles_only = l.encode(cfg_triangles_only()).unwrap();
        assert_eq!((with_outlines.len(), triangles_only.len()), (82, 68));
    }

    #[rstest]
    #[case::lines_and_polygons(vec![
        line(&[(0, 0), (10, 10), (20, 0)]),
        Geometry::Polygon(square(40, 40)),
    ])]
    #[case::points_and_polygons(vec![pt(5, 5), Geometry::Polygon(square(20, 20))])]
    fn a_layer_with_more_than_polygons_keeps_its_outlines(#[case] geoms: Vec<Geometry<i32>>) {
        let l = layer(geoms, None, &[]);
        let bytes = l.encode(cfg_triangles_only()).unwrap();
        assert_eq!(
            header_bits(&bytes).last().map(String::as_str),
            Some("geometry layout = TessPolygonsWithOutlines"),
        );
        assert_differential_with(&l, cfg_tessellated().with_triangles_only(true));
    }

    #[test]
    fn v1_ignores_triangles_only() {
        let l = layer(vec![Geometry::Polygon(square(0, 0))], None, &[]);
        let v1 = cfg_tessellated();
        assert_eq!(
            l.encode(v1.with_triangles_only(true)).unwrap(),
            l.encode(v1).unwrap(),
        );
    }

    #[test]
    fn a_point_among_bare_triangles_is_rejected() {
        let l = layer(vec![Geometry::Polygon(square(0, 0))], None, &[]);
        let mut bytes = l.encode(cfg_triangles_only()).unwrap();
        let header = annotate(&bytes)
            .regions
            .iter()
            .find(|r| r.label == "header")
            .expect("a header region")
            .offset;
        let uniform_point = 1 << 4;
        bytes[header] = (bytes[header] & 0b1000_1111) | uniform_point;
        let layers = Parser::default().parse_layers(&bytes).expect("parse");
        let err = layers
            .into_iter()
            .next()
            .unwrap()
            .into_tile(&mut Decoder::default())
            .expect_err("a point has no triangles");
        assert_snapshot!(err, @"geometry[0]: Point requires outlines, which a triangles-only layer does not store");
    }
    const TRIANGLES_ONLY: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test/synthetic/0x02/mix_2_poly_mpoly_tri.mlt"
    ));

    fn decode_patched(offset: usize, from: u8, to: u8) -> String {
        let mut bytes = TRIANGLES_ONLY.to_vec();
        assert_eq!(bytes[offset], from);
        bytes[offset] = to;
        let layers = Parser::default().parse_layers(&bytes).expect("parse");
        let layer = layers.into_iter().next().unwrap();
        layer
            .into_tile(&mut Decoder::default())
            .unwrap_err()
            .to_string()
    }

    #[test]
    fn a_tree_cycle_naming_a_missing_vertex_is_rejected() {
        assert_snapshot!(decode_patched(0x2f, 11, 12), @"triangle tree names vertex 12 of a layer with 12 vertices");
    }

    #[test]
    fn triangle_lengths_not_summing_to_the_symbols_are_rejected() {
        assert_snapshot!(decode_patched(0x13, 7, 6), @"triangle trees are malformed: triangle lengths do not sum to the symbol count");
    }
}

mod strings {
    use super::*;

    /// A value long and repetitive enough for FSST to pay off its symbol table.
    /// The seed leads, so two of them share no prefix worth coding.
    fn long(seed: usize) -> String {
        let lead = char::from(b'a' + u8::try_from(seed).unwrap());
        format!("{lead}_residential_zone_north_sector_").repeat(16)
    }

    /// As [`long`], with the seed last, so two of them share all but their final bytes.
    fn long_shared(seed: usize) -> String {
        format!("residential_zone_north_sector_{seed:03}_").repeat(16)
    }

    /// Distinct short values sharing no prefix, which no dictionary or symbol table improves on.
    fn plain_values(n: usize) -> Vec<String> {
        (0..n)
            .map(|i| {
                let lead = char::from(b'a' + u8::try_from(i).unwrap());
                format!("{lead}_zone_marker")
            })
            .collect()
    }

    /// Two long values alternating, which is what a dictionary is for.
    /// They share no prefix, so front coding would only add a length per entry.
    fn dict_values(n: usize) -> Vec<String> {
        (0..n)
            .map(|i| {
                let lead = if i % 2 == 0 { "A" } else { "B" };
                lead.repeat(30)
            })
            .collect()
    }

    /// As [`dict_values`], with the two sharing all but their last byte, which front coding factors out.
    fn front_dict_values(n: usize) -> Vec<String> {
        (0..n)
            .map(|i| format!("{}{}", "A".repeat(30), i % 2))
            .collect()
    }

    /// Distinct values FSST compresses, so a dictionary of them would only add codes.
    fn fsst_values(n: usize) -> Vec<String> {
        (0..n).map(long).collect()
    }

    /// Four of those repeated, so the dictionary and the symbol table both pay off.
    fn fsst_dict_values(n: usize) -> Vec<String> {
        (0..n).map(|i| long(i % 4)).collect()
    }

    /// As [`fsst_dict_values`], with the four sharing prefixes for front coding to factor out.
    fn fsst_front_dict_values(n: usize) -> Vec<String> {
        (0..n).map(|i| long_shared(i % 4)).collect()
    }

    type Values = fn(usize) -> Vec<String>;

    fn column(values: impl IntoIterator<Item = Option<String>>) -> TileLayer {
        let values: Vec<PropValue> = values
            .into_iter()
            .map(|v| PropValue::Str(v.map(Into::into)))
            .collect();
        layer(points(&"1".repeat(values.len())), None, &[("v", values)])
    }

    /// How many of the tile's dictionary blobs are front coded.
    fn front_coded_blobs(bytes: &[u8]) -> usize {
        dump_text(bytes).matches("logical = FrontCoded").count()
    }

    /// The layout of every string column in the tile, in wire order.
    fn layouts(bytes: &[u8]) -> Vec<String> {
        dump_fields(bytes, "string layout = ")
    }

    #[rstest]
    #[case::plain(plain_values as Values, "Plain", false)]
    #[case::dict(dict_values as Values, "Dict", false)]
    #[case::front_dict(front_dict_values as Values, "Dict", false)]
    #[case::fsst(fsst_values as Values, "Fsst", true)]
    #[case::fsst_dict(fsst_dict_values as Values, "FsstDict", true)]
    #[case::fsst_front_dict(fsst_front_dict_values as Values, "FsstDict", true)]
    fn each_layout_round_trips(#[case] values: Values, #[case] layout: &str, #[case] fsst: bool) {
        let l = column(values(9).into_iter().map(Some));
        assert_differential(&l);
        let bytes = l.encode(cfg_v2().with_fsst(fsst)).unwrap();
        assert_eq!(layouts(&bytes), [layout]);
    }

    #[rstest]
    #[case::plain(plain_values as Values, "Plain", false)]
    #[case::dict(dict_values as Values, "Dict", false)]
    #[case::front_dict(front_dict_values as Values, "Dict", false)]
    #[case::fsst(fsst_values as Values, "Fsst", true)]
    #[case::fsst_dict(fsst_dict_values as Values, "FsstDict", true)]
    #[case::fsst_front_dict(fsst_front_dict_values as Values, "FsstDict", true)]
    fn each_layout_round_trips_with_nulls(
        #[case] values: Values,
        #[case] layout: &str,
        #[case] fsst: bool,
    ) {
        let l = column(
            values(12)
                .into_iter()
                .enumerate()
                .map(|(i, v)| (i % 3 != 0).then_some(v)),
        );
        assert_differential(&l);
        let bytes = l.encode(cfg_v2().with_fsst(fsst)).unwrap();
        assert_eq!(layouts(&bytes), [layout]);
    }

    #[rstest]
    #[case::plain(plain_values as Values)]
    #[case::dict(dict_values as Values)]
    #[case::front_dict(front_dict_values as Values)]
    fn fsst_over_front_coded_suffixes_beats_the_layout_that_wins_without_it(
        #[case] values: Values,
    ) {
        let l = column(values(9).into_iter().map(Some));
        let with = l.encode(cfg_v2()).unwrap();
        let without = l.encode(cfg_v2().with_fsst(false)).unwrap();
        assert_eq!(layouts(&with), ["FsstDict"]);
        assert!(
            with.len() < without.len(),
            "{} < {}",
            with.len(),
            without.len()
        );
    }

    #[rstest]
    #[case::entries_share_a_prefix(front_dict_values as Values, 1)]
    #[case::entries_share_no_prefix(dict_values as Values, 0)]
    fn front_coding_wins_on_shared_prefixes(#[case] values: Values, #[case] front_coded: usize) {
        let l = column(values(9).into_iter().map(Some));
        assert_differential(&l);
        let bytes = l.encode(cfg_v2().with_fsst(false)).unwrap();
        assert_eq!(front_coded_blobs(&bytes), front_coded);
    }

    #[test]
    fn empty_values_round_trip() {
        let l = column(["", "a", "", "bb", ""].map(|v| Some(v.to_string())));
        assert_differential(&l);
    }

    #[test]
    fn a_column_starting_with_a_null_round_trips() {
        let l = column([None, Some("a".to_string()), None, Some("b".to_string())]);
        assert_differential(&l);
    }

    #[test]
    fn non_ascii_values_round_trip() {
        let l = column(["日本語", "Ünïcödé", "🌍 🌏", ""].map(|v| Some(v.to_string())));
        assert_differential(&l);
    }

    #[test]
    fn a_column_of_one_repeated_value_round_trips() {
        let l = column((0..8).map(|_| Some("same".to_string())));
        assert_differential(&l);
    }

    /// A layer whose two string columns hold the same values, which both versions group.
    fn shared_dict_layer() -> TileLayer {
        let shared: Vec<PropValue> = dict_values(6)
            .into_iter()
            .map(|v| PropValue::Str(Some(v.into())))
            .collect();
        layer(
            points(&"1".repeat(6)),
            None,
            &[("a", shared.clone()), ("b", shared)],
        )
    }

    /// The base type of every v1 column in the tile, in wire order.
    fn base_types(bytes: &[u8]) -> Vec<String> {
        dump_fields(bytes, "base type = ")
    }

    /// The data type of every v2 column in the tile, in wire order.
    fn data_types(bytes: &[u8]) -> Vec<String> {
        dump_fields(bytes, "data type = ")
    }

    #[test]
    fn columns_v1_would_share_a_dictionary_share_one_in_v2_too() {
        let l = shared_dict_layer();
        assert_eq!(
            base_types(&l.encode(cfg_v1()).unwrap()),
            ["Geometry", "SharedDict", "Str", "Str"]
        );
        assert_eq!(
            data_types(&l.encode(cfg_v2()).unwrap()),
            ["SharedDict", "Str", "Str"]
        );
        assert_differential(&l);
    }

    #[test]
    fn a_child_count_larger_than_the_bytes_left_is_rejected() {
        let bytes = shared_dict_layer().encode(cfg_v2()).unwrap();
        let tree = annotate(&bytes);
        let count = tree
            .regions
            .iter()
            .find(|r| r.label == "child_count")
            .expect("a child_count region");
        assert_eq!((count.len, bytes[count.offset]), (1, 2));
        let remaining = bytes.len() - count.offset - 1;
        let bytes = splice_layer(&bytes, count.offset, &[0xfb, 0xbf, 0xfb, 0x08]);
        let err = Parser::default()
            .parse_layers(&bytes)
            .expect_err("a rejection");
        assert_eq!(
            err.to_string(),
            format!("buffer underflow: needed 18800635 bytes, but only {remaining} remain")
        );
    }

    /// Replace the byte at `offset` with `with`, re-sizing the layer's length prefix to match.
    fn splice_layer(bytes: &[u8], offset: usize, with: &[u8]) -> Vec<u8> {
        let size_len = bytes
            .iter()
            .position(|b| b & 0x80 == 0)
            .expect("a size varint")
            + 1;
        let mut body = bytes[size_len..].to_vec();
        let at = offset - size_len;
        body.splice(at..=at, with.iter().copied());
        let mut size = body.len();
        let mut out = Vec::new();
        while size >= 0x80 {
            out.push(u8::try_from(size & 0x7f).expect("seven bits") | 0x80);
            size >>= 7;
        }
        out.push(u8::try_from(size).expect("seven bits"));
        out.extend_from_slice(&body);
        out
    }

    #[test]
    fn a_shared_dictionary_is_smaller_than_per_column_ones() {
        let l = shared_dict_layer();
        let shared = l.encode(cfg_v2()).unwrap().len();
        let separate = l.encode(cfg_v2().with_shared_dict(false)).unwrap().len();
        assert!(shared < separate, "shared {shared} vs separate {separate}");
    }

    #[test]
    fn a_shared_dictionary_with_nulls_round_trips() {
        let mask = "101101";
        let values = |offset: usize| -> Vec<PropValue> {
            mask.bytes()
                .enumerate()
                .map(|(i, b)| {
                    PropValue::Str((b == b'1').then(|| format!("name:{}", (i + offset) % 3).into()))
                })
                .collect()
        };
        let l = layer(
            points(mask),
            None,
            &[("name:de", values(0)), ("name:en", values(1))],
        );
        assert_eq!(
            data_types(&l.encode(cfg_v2()).unwrap()),
            ["SharedDict", "Str", "Str"]
        );
        assert_differential(&l);
    }

    #[test]
    fn every_shared_dictionary_child_carries_its_own_presence() {
        let values = dict_values(6);
        let child = |mask: &str| -> Vec<PropValue> {
            mask.bytes()
                .zip(&values)
                .map(|(b, v)| PropValue::Str((b == b'1').then(|| v.as_str().into())))
                .collect()
        };
        let l = layer(
            points("111111"),
            None,
            &[("name:0", child("101101")), ("name:1", child("101101"))],
        );
        assert_differential(&l);
        let dump = dump_text(&l.encode(cfg_v2()).unwrap());
        assert_eq!(dump.matches("presence = Inline").count(), 2, "{dump}");
    }
}

mod floats {
    use super::*;

    fn f64_col(values: &[f64]) -> Vec<PropValue> {
        values.iter().map(|&v| PropValue::F64(Some(v))).collect()
    }

    fn column(values: &[f64]) -> TileLayer {
        let mask = "1".repeat(values.len());
        layer(points(&mask), None, &[("v", f64_col(values))])
    }

    /// The column's values as bit patterns, so NaN and `-0.0` compare usefully.
    #[expect(
        clippy::wildcard_enum_match_arm,
        reason = "only float values are expected"
    )]
    fn value_bits(tile: &TileLayer) -> Vec<u64> {
        tile.features()
            .iter()
            .map(|f| match f.properties()[0] {
                PropValue::F64(Some(v)) => v.to_bits(),
                PropValue::F32(Some(v)) => u64::from(v.to_bits()),
                ref other => panic!("unexpected {other:?}"),
            })
            .collect()
    }

    #[test]
    fn floats_keep_every_bit() {
        let values = [0.0, -0.0, f64::NAN, f64::INFINITY, 1.5, -2.25e300];
        let l = column(&values);
        let expected: Vec<u64> = values.iter().map(|v| v.to_bits()).collect();
        for cfg in [cfg_v1(), cfg_v2()] {
            let (_, tile) = decode(&l.encode(cfg).unwrap());
            assert_eq!(value_bits(&tile), expected);
        }
    }

    #[test]
    fn f32_columns_round_trip() {
        let values: Vec<PropValue> = (0..12)
            .map(|i| PropValue::F32(Some([1.5_f32, 2.5, 3.5][i % 3])))
            .collect();
        let l = layer(points(&"1".repeat(12)), None, &[("v", values)]);
        assert_differential(&l);
    }

    #[test]
    fn an_optional_float_column_round_trips() {
        let values: Vec<PropValue> = (0..12)
            .map(|i| PropValue::F64((i % 3 != 0).then(|| f64::from(i % 2) + 0.5)))
            .collect();
        let l = layer(points(&"1".repeat(12)), None, &[("v", values)]);
        assert_differential(&l);
    }
}

mod presence {
    use super::*;

    /// A column present on a single run of features, which byte-RLE codes in a few bytes.
    fn one_run(n: usize) -> String {
        (0..n)
            .map(|i| if (64..96).contains(&i) { '1' } else { '0' })
            .collect()
    }

    #[test]
    fn the_web_profile_stores_presence_as_a_bitmap() {
        let mask = one_run(256);
        let l = layer(points(&"1".repeat(256)), None, &[("v", opt_col(&mask))]);
        let web = EncoderConfig::from_profile(Profile::Web(Network::Cellular4G));
        assert_differential_with(&l, web);
        let dump = dump_text(&l.encode(web.with_wire_version(WireVersion::V02)).unwrap());
        assert!(dump.contains("presence = Inline(Bitmap)"), "{dump}");
    }

    #[test]
    fn the_uncompressed_profile_codes_a_run_of_presence_byte_rle() {
        let mask = one_run(256);
        let l = layer(points(&"1".repeat(256)), None, &[("v", opt_col(&mask))]);
        let offline = EncoderConfig::from_profile(Profile::Web(Network::Offline));
        assert_differential_with(&l, offline);
        let offline = offline.with_wire_version(WireVersion::V02);
        let dump = dump_text(&l.encode(offline).unwrap());
        assert!(dump.contains("presence = Inline(ByteRle)"), "{dump}");
        let bitmap = l.encode(offline.with_presence_rle(false)).unwrap();
        assert!(l.encode(offline).unwrap().len() < bitmap.len());
    }

    #[test]
    fn a_bool_column_codes_its_values_byte_rle_when_allowed() {
        let values: Vec<PropValue> = (0..256)
            .map(|i| PropValue::Bool(Some((64..96).contains(&i))))
            .collect();
        let l = layer(points(&"1".repeat(256)), None, &[("b", values)]);
        assert_differential_with(&l, cfg_v1().with_presence_rle(true));
        let cfg = cfg_v2().with_presence_rle(true);
        assert!(
            l.encode(cfg).unwrap().len() < l.encode(cfg.with_presence_rle(false)).unwrap().len()
        );
    }
}

mod predicted_vertices {
    use mlt_core::test_helpers::stream_logicals;
    use mlt_core::wire::{DictionaryType, LogicalEncoding, StreamType, VertexLogical};

    use super::*;

    fn cfg_predicted() -> EncoderConfig {
        cfg_v2()
            .with_predicted_vertices(true)
            .with_vertex_dict(false)
    }

    fn vertex_logicals(bytes: &[u8]) -> Vec<LogicalEncoding> {
        stream_logicals(bytes, StreamType::Data(DictionaryType::Vertex))
    }

    /// Pairs of polygons whose second walks the border of the first backwards, as
    /// neighbouring land-use polygons do, so half the vertices repeat earlier ones.
    fn block() -> Vec<Geometry<i32>> {
        (0..40)
            .flat_map(|i| {
                let border: Vec<(i32, i32)> = (0..16)
                    .map(|k| (i * 60 + k * 7 + (k * k) % 13, i * 20 + k * 3 - (k * 5) % 11))
                    .collect();
                let mut back = border.clone();
                back.reverse();
                [border, back].map(|r| Geometry::Polygon(Polygon::new(ring(&r), vec![])))
            })
            .collect()
    }

    #[test]
    fn shared_borders_are_predicted_and_decode_unchanged() {
        let l = layer(block(), None, &[]);
        let predicted = l.encode(cfg_predicted()).unwrap();
        assert_eq!(
            vertex_logicals(&predicted),
            [LogicalEncoding::Vertex(VertexLogical::Predicted)]
        );
        let plain = l
            .encode(cfg_predicted().with_predicted_vertices(false))
            .unwrap();
        assert!(
            predicted.len() < plain.len(),
            "{} vs {}",
            predicted.len(),
            plain.len()
        );
        assert_eq!(decode(&predicted).1, decode(&plain).1);
        assert_dump_covers(&predicted);
    }

    #[rstest]
    fn a_tessellated_layer_predicts_only_with_outlines(
        #[values(false, true)] triangles_only: bool,
    ) {
        let l = layer(block(), None, &[]);
        let cfg = cfg_predicted()
            .with_tessellation(Tessellation::Always)
            .with_triangles_only(triangles_only);
        let bytes = l.encode(cfg).unwrap();
        let expected = if triangles_only {
            VertexLogical::ComponentwiseDelta
        } else {
            VertexLogical::Predicted
        };
        assert_eq!(vertex_logicals(&bytes), [LogicalEncoding::Vertex(expected)]);
        assert_eq!(
            decode(&bytes).1,
            decode(&l.encode(cfg.with_predicted_vertices(false)).unwrap()).1
        );
    }

    #[test]
    fn v1_ignores_predicted_vertices() {
        let l = layer(block(), None, &[]);
        assert_eq!(
            l.encode(cfg_v1().with_predicted_vertices(true)).unwrap(),
            l.encode(cfg_v1()).unwrap()
        );
    }
}
