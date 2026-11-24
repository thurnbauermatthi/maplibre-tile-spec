//! `WGS84` geodesy, 4x4 column-major transforms, and `sRGB` colour packing for `glTF` meshes.

/// `WGS84` semi-major axis in metres.
const WGS84_A: f64 = 6_378_137.0;
/// `WGS84` first eccentricity squared, `f * (2 - f)` for `1 / f = 298.257223563`.
const WGS84_E2: f64 = 0.006_694_379_990_141_316_5;
/// `WGS84` semi-minor axis in metres, `a * sqrt(1 - e^2)`.
const WGS84_B: f64 = 6_356_752.314_245_179;
/// `WGS84` second eccentricity squared, `(a^2 - b^2) / b^2`.
const WGS84_EP2: f64 = (WGS84_A * WGS84_A - WGS84_B * WGS84_B) / (WGS84_B * WGS84_B);
/// Minimum distance from the Earth's centre (in metres) accepted for an `ECEF` coordinate.
pub(super) const MIN_ECEF_RADIUS: f64 = 6_000_000.0;

/// 4x4 column-major affine matrix, matching `glTF` node matrices.
pub(super) type Mat4 = [f64; 16];

pub(super) const IDENTITY: Mat4 = [
    1.0, 0.0, 0.0, 0.0, //
    0.0, 1.0, 0.0, 0.0, //
    0.0, 0.0, 1.0, 0.0, //
    0.0, 0.0, 0.0, 1.0,
];

/// Rotates `glTF` Y-up `(x, y, z)` to `ECEF` Z-up `(x, -z, y)`.
pub(super) const Y_UP_TO_Z_UP: Mat4 = [
    1.0, 0.0, 0.0, 0.0, //
    0.0, 0.0, 1.0, 0.0, //
    0.0, -1.0, 0.0, 0.0, //
    0.0, 0.0, 0.0, 1.0,
];

pub(super) fn translation([x, y, z]: [f64; 3]) -> Mat4 {
    [
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        x, y, z, 1.0,
    ]
}

/// Builds a column-major 4x4 matrix from `glTF` `translation`, quaternion `rotation` `[x, y, z, w]` and `scale`.
pub(super) fn from_trs(trans: [f64; 3], rot: [f64; 4], scale: [f64; 3]) -> Mat4 {
    let [qx, qy, qz, qw] = rot;
    let len2 = qx * qx + qy * qy + qz * qz + qw * qw;
    let (rx, ry, rz, rw) = if len2 > 0.0 {
        let inv = 1.0 / len2.sqrt();
        (qx * inv, qy * inv, qz * inv, qw * inv)
    } else {
        (0.0, 0.0, 0.0, 1.0)
    };
    let (x2, y2, z2) = (rx + rx, ry + ry, rz + rz);
    let (xx, xy, xz) = (rx * x2, rx * y2, rx * z2);
    let (yy, yz, zz) = (ry * y2, ry * z2, rz * z2);
    let (wx, wy, wz) = (rw * x2, rw * y2, rw * z2);
    [
        (1.0 - (yy + zz)) * scale[0],
        (xy + wz) * scale[0],
        (xz - wy) * scale[0],
        0.0,
        (xy - wz) * scale[1],
        (1.0 - (xx + zz)) * scale[1],
        (yz + wx) * scale[1],
        0.0,
        (xz + wy) * scale[2],
        (yz - wx) * scale[2],
        (1.0 - (xx + yy)) * scale[2],
        0.0,
        trans[0],
        trans[1],
        trans[2],
        1.0,
    ]
}

pub(super) fn mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = [0.0; 16];
    for col in 0..4 {
        for row in 0..4 {
            out[col * 4 + row] = (0..4).map(|k| a[k * 4 + row] * b[col * 4 + k]).sum();
        }
    }
    out
}

pub(super) fn apply(m: &Mat4, [x, y, z]: [f64; 3]) -> [f64; 3] {
    [
        m[0] * x + m[4] * y + m[8] * z + m[12],
        m[1] * x + m[5] * y + m[9] * z + m[13],
        m[2] * x + m[6] * y + m[10] * z + m[14],
    ]
}

/// Determinant of the upper-left 3x3 linear block of `m` (negative when `m` mirrors handedness).
pub(super) fn det3(m: &Mat4) -> f64 {
    m[0] * (m[5] * m[10] - m[9] * m[6]) - m[4] * (m[1] * m[10] - m[9] * m[2])
        + m[8] * (m[1] * m[6] - m[5] * m[2])
}

pub(super) fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub(super) fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub(super) fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(super) fn norm(v: [f64; 3]) -> f64 {
    dot(v, v).sqrt()
}

/// Local east-north-up tangent frame at a `WGS84` anchor, mapping `glTF` metres (`+X` east, `+Y` up, `+Z` south) to geodetic radians and height.
#[derive(Clone, Copy, Debug)]
pub(super) struct Frame {
    lon: f64,
    lat: f64,
    east_radius: f64,
    north_radius: f64,
}

impl Frame {
    pub(super) fn new([lon_deg, lat_deg]: [f64; 2]) -> Self {
        let (lon, lat) = (lon_deg.to_radians(), lat_deg.to_radians());
        let sin = lat.sin();
        let w2 = 1.0 - WGS84_E2 * sin * sin;
        let prime_vertical = WGS84_A / w2.sqrt();
        Self {
            lon,
            lat,
            east_radius: prime_vertical * lat.cos(),
            north_radius: prime_vertical * (1.0 - WGS84_E2) / w2,
        }
    }

    /// Maps `glTF` `[east, up, south]` in metres to `[lon_rad, lat_rad, height_m]`.
    pub(super) fn to_geodetic(self, [east, up, south]: [f64; 3]) -> [f64; 3] {
        [
            self.lon + east / self.east_radius,
            self.lat - south / self.north_radius,
            up,
        ]
    }
}

/// Bowring's closed-form `ECEF` `[x, y, z]` (metres) to `WGS84` `[lon_rad, lat_rad, height_m]`.
pub(super) fn ecef_to_geodetic([ex, ey, ez]: [f64; 3]) -> [f64; 3] {
    let rho = ex.hypot(ey);
    let theta = (ez * WGS84_A).atan2(rho * WGS84_B);
    let (st, ct) = theta.sin_cos();
    let lat = (ez + WGS84_EP2 * WGS84_B * st.powi(3)).atan2(rho - WGS84_E2 * WGS84_A * ct.powi(3));
    let (sl, cl) = lat.sin_cos();
    let prime_vertical = WGS84_A / (1.0 - WGS84_E2 * sl * sl).sqrt();
    let height = if cl.abs() > 1e-10 {
        rho / cl - prime_vertical
    } else {
        ez.abs() - WGS84_B
    };
    [ey.atan2(ex), lat, height]
}

/// Unit `ECEF` surface normal at geodetic `[lon_rad, lat_rad]`.
pub(super) fn ecef_up([lon, lat]: [f64; 2]) -> [f64; 3] {
    let (slon, clon) = lon.sin_cos();
    let (slat, clat) = lat.sin_cos();
    [clat * clon, clat * slon, slat]
}

/// Packs linear `RGB` `[r, g, b]` in `0.0..=1.0` into `0xRRGGBB` `sRGB` (IEC 61966-2-1).
pub(super) fn srgb([red, green, blue]: [f32; 3]) -> u32 {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped to 0.0..=1.0 before scaling to 0..=255"
    )]
    let byte = |chan: f32| {
        let clamped = chan.clamp(0.0, 1.0);
        let encoded = if clamped <= 0.003_130_8 {
            12.92 * clamped
        } else {
            1.055 * clamped.powf(1.0 / 2.4) - 0.055
        };
        (encoded * 255.0).round() as u32
    };
    (byte(red) << 16) | (byte(green) << 8) | byte(blue)
}
