import { GEOMETRY_TYPE } from "./geometry/geometryType";
import { classifyRings } from "./geometry/classifyRings";
import type { Geometry } from "./geometry/geometryVector";
import type FeatureTable from "./featureTable";

/**
 * Converts decoded feature tables into the GeoJSON shape the synthetic fixtures are written in.
 *
 * Shared by the decoder's synthetic tests and the encoder's round trip, so both compare against the
 * fixtures in exactly the same way. The layer name and extent are carried in the properties, since
 * GeoJSON has nowhere else to put them.
 */
export function featureTablesToFeatureCollection(featureTables: FeatureTable[]): GeoJSON.FeatureCollection {
    const features: GeoJSON.Feature[] = [];
    for (const table of featureTables) {
        for (const feature of table.getFeatures()) {
            const geojsonFeature: GeoJSON.Feature = {
                type: "Feature",
                geometry: getGeometry(feature.geometry),
                properties: {
                    _layer: table.name,
                    _extent: table.extent,
                    ...Object.fromEntries(Object.entries(feature.properties).map(([k, v]) => [k, safeNumber(v)])),
                },
            };
            if (feature.id !== undefined) {
                geojsonFeature.id = Number(feature.id);
            }
            features.push(geojsonFeature);
        }
    }
    return { type: "FeatureCollection", features };
}

function groupRingsByCount(coords: number[][][], counts: number[]): number[][][][] {
    const polys: number[][][][] = new Array(counts.length);
    let r = 0;
    for (let p = 0; p < counts.length; p++) {
        const count = counts[p];
        if (count === 0) {
            polys[p] = coords[r] && coords[r].length === 0 ? [coords[r++]] : [[]];
        } else {
            polys[p] = coords.slice(r, r + count);
            r += count;
        }
    }
    return polys;
}

/**
 * Converts one decoded geometry to GeoJSON. Multi-polygons need their flat ring list grouped back
 * into polygons, which {@link classifyRings} does from the winding order when explicit ring counts
 * are absent.
 */
export function getGeometry(geometry: Geometry): GeoJSON.Geometry {
    const coords = geometry.coordinates.map((ring) => ring.map((p) => [p.x, p.y]));
    if (geometry.trianglesOnly) return { type: "MultiPolygon", coordinates: coords.map((ring) => [ring]) };

    switch (geometry.type) {
        case GEOMETRY_TYPE.POINT:
            return { type: "Point", coordinates: coords[0][0] };
        case GEOMETRY_TYPE.LINESTRING:
            return { type: "LineString", coordinates: coords[0] };
        case GEOMETRY_TYPE.POLYGON:
            return { type: "Polygon", coordinates: coords };
        case GEOMETRY_TYPE.MULTIPOINT:
            return { type: "MultiPoint", coordinates: coords.map((r) => r[0]) };
        case GEOMETRY_TYPE.MULTILINESTRING:
            return { type: "MultiLineString", coordinates: coords };
        case GEOMETRY_TYPE.MULTIPOLYGON:
            return {
                type: "MultiPolygon",
                coordinates: geometry.ringsPerPolygon
                    ? groupRingsByCount(coords, geometry.ringsPerPolygon)
                    : classifyRings(coords),
            };
        default:
            throw new Error(`Unsupported geometry type: ${geometry.type}`);
    }
}

function safeNumber<T>(val: bigint | T): unknown {
    if (typeof val === "bigint") return Number(val);
    if (Array.isArray(val)) return val.map(safeNumber);
    if (val !== null && typeof val === "object") {
        return Object.fromEntries(Object.entries(val).map(([k, v]) => [k, safeNumber(v)]));
    }
    return val;
}
