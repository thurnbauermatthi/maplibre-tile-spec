import FeatureTable from "./vector/featureTable";
import { type Column, ScalarType } from "./metadata/tileset/tilesetMetadata";
import IntWrapper from "./decoding/intWrapper";
import { decodeStreamMetadata, type RleEncodedStreamMetadata } from "./metadata/tile/streamMetadataDecoder";
import { VectorType } from "./vector/vectorType";
import { Int32FlatVector } from "./vector/flat/int32FlatVector";
import BitVector from "./vector/flat/bitVector";
import { BooleanFlatVector } from "./vector/flat/booleanFlatVector";
import { FloatFlatVector } from "./vector/flat/floatFlatVector";
import { ObjectFlatVector } from "./vector/flat/objectFlatVector";
import {
    decodeUnsignedConstInt32Stream,
    decodeUnsignedConstInt64Stream,
    decodeUnsignedInt64AsFloat64Stream,
    decodeUnsignedInt32Stream,
    decodeUnsignedInt64Stream,
    decodeSequenceInt32Stream,
    decodeSequenceInt64Stream,
    getVectorType,
} from "./decoding/integerStreamDecoder";
import { Int32SequenceVector } from "./vector/sequence/int32SequenceVector";
import { Int64FlatVector } from "./vector/flat/int64FlatVector";
import { Int64SequenceVector } from "./vector/sequence/int64SequenceVector";
import type { IdVector } from "./vector/idVector";
import { decodeVarintInt32 } from "./decoding/integerDecodingUtils";
import { decodeGeometryColumn } from "./decoding/geometryDecoder";
import { decodePropertyColumn } from "./decoding/propertyDecoder";
import { Int32ConstVector } from "./vector/constant/int32ConstVector";
import { Int64ConstVector } from "./vector/constant/int64ConstVector";
import type GeometryScaling from "./decoding/geometryScaling";
import { decodeBooleanRle } from "./decoding/decodingUtils";
import { DoubleFlatVector } from "./vector/flat/doubleFlatVector";
import { decodeEmbeddedTileSetMetadata } from "./metadata/tileset/embeddedTilesetMetadataDecoder";
import { hasStreamCount, isGeometryColumn, isLogicalIdColumn } from "./metadata/tileset/typeMap";
import type { StreamMetadata } from "./metadata/tile/streamMetadataDecoder";
import type { GeometryVector } from "./vector/geometry/geometryVector";
import { ConstGeometryVector } from "./vector/geometry/constGeometryVector";
import { FlatGeometryVector } from "./vector/geometry/flatGeometryVector";
import { createConstGpuVector } from "./vector/geometry/constGpuVector";
import { createFlatGpuVector } from "./vector/geometry/flatGpuVector";
import { GEOMETRY_TYPE } from "./vector/geometry/geometryType";
import type { TopologyVector } from "./vector/geometry/topologyVector";
import { VertexBufferType } from "./vector/geometry/vertexBufferType";
import Vector from "./vector/vector";
import type { GpuVector } from "./vector/geometry/gpuVector";
import { resetBudget } from "./decoding/v2/binary";
import { type V2Layer, type V2PropertyColumn, decodeSingleLayer02 } from "./decoding/v2/layer";

/**
 * Decodes a tile with embedded metadata (Tag 0x01 and Tag 0x02 formats).
 * This is the primary decoder function for MLT tiles.
 *
 * @param tile The tile data to decode (will be decompressed if gzip-compressed)
 * @param geometryScaling Optional geometry scaling parameters
 * @param idWithinMaxSafeInteger If true, limits ID values to JavaScript safe integer range (53 bits)
 */
export default function decodeTile(
    tile: Uint8Array,
    geometryScaling?: GeometryScaling,
    idWithinMaxSafeInteger = true,
): FeatureTable[] {
    resetBudget();
    const offset = new IntWrapper(0);
    const featureTables: FeatureTable[] = [];

    while (offset.get() < tile.length) {
        const blockLength = decodeVarintInt32(tile, offset, 1)[0] >>> 0;
        const blockStart = offset.get();
        const blockEnd = blockStart + blockLength;
        if (blockLength === 0 || blockEnd > tile.length) {
            throw new RangeError(`Block overruns tile or has zero size: ${blockEnd} > ${tile.length}`);
        }

        const tag = decodeVarintInt32(tile, offset, 1)[0] >>> 0;
        if (offset.get() > blockEnd) {
            throw new RangeError("Layer tag overruns block");
        }
        if (tag === 2) {
            const layer = decodeSingleLayer02(tile.subarray(offset.get(), blockEnd), undefined, false);
            featureTables.push(v2LayerToFeatureTable(layer, geometryScaling, idWithinMaxSafeInteger));
            offset.set(blockEnd);
            continue;
        }
        if (tag !== 1) {
            // Skip unknown block types
            offset.set(blockEnd);
            continue;
        }

        const [metadata, extent] = decodeEmbeddedTileSetMetadata(tile, offset);
        const featureTableMetadata = metadata.featureTables[0];

        let idVector: IdVector | null = null;
        let geometryVector: GeometryVector | GpuVector | null = null;
        const propertyVectors: Vector[] = [];
        let numFeatures = 0;

        for (const columnMetadata of featureTableMetadata.columns) {
            const columnName = columnMetadata.name;

            if (isLogicalIdColumn(columnMetadata)) {
                let nullabilityBuffer = null;
                // Check column metadata nullable flag, not numStreams (ID columns don't have stream count)
                if (columnMetadata.nullable) {
                    const presentStreamMetadata = decodeStreamMetadata(tile, offset);
                    const streamDataStart = offset.get();
                    const values = decodeBooleanRle(
                        tile,
                        presentStreamMetadata.numValues,
                        presentStreamMetadata.byteLength,
                        offset,
                    );
                    offset.set(streamDataStart + presentStreamMetadata.byteLength);
                    nullabilityBuffer = new BitVector(values, presentStreamMetadata.numValues);
                }

                const idDataStreamMetadata = decodeStreamMetadata(tile, offset);
                // decompressedCount is the count WITHOUT nulls, but we may have nulls
                numFeatures = nullabilityBuffer ? nullabilityBuffer.size() : idDataStreamMetadata.decompressedCount;

                idVector = decodeIdColumn(
                    tile,
                    columnMetadata,
                    offset,
                    columnName,
                    idDataStreamMetadata,
                    nullabilityBuffer ?? numFeatures,
                    idWithinMaxSafeInteger,
                );
            } else if (isGeometryColumn(columnMetadata)) {
                const numStreams = decodeVarintInt32(tile, offset, 1)[0];

                // If no ID column, get numFeatures from geometry type stream metadata
                if (numFeatures === 0) {
                    const savedOffset = offset.get();
                    const geometryTypeMetadata = decodeStreamMetadata(tile, offset);
                    numFeatures = geometryTypeMetadata.decompressedCount;
                    offset.set(savedOffset); // Reset to re-read in decodeGeometryColumn
                }

                if (geometryScaling) {
                    geometryScaling.scale = geometryScaling.extent / extent;
                }

                geometryVector = decodeGeometryColumn(tile, numStreams, offset, numFeatures, geometryScaling);
            } else {
                const columnHasStreamCount = hasStreamCount(columnMetadata);
                const numStreams = columnHasStreamCount ? decodeVarintInt32(tile, offset, 1)[0] : 1;

                if (numStreams === 0) {
                    continue;
                }

                const propertyVector = decodePropertyColumn(
                    tile,
                    offset,
                    columnMetadata,
                    numStreams,
                    numFeatures,
                    undefined,
                );
                if (propertyVector) {
                    if (Array.isArray(propertyVector)) {
                        for (const property of propertyVector) {
                            propertyVectors.push(property);
                        }
                    } else {
                        propertyVectors.push(propertyVector);
                    }
                }
            }
        }

        const featureTable = new FeatureTable(
            featureTableMetadata.name,
            geometryVector,
            idVector,
            propertyVectors,
            extent,
        );
        featureTables.push(featureTable);
        offset.set(blockEnd);
    }

    return featureTables;
}

function identityOffsets(count: number): Uint32Array {
    const out = new Uint32Array(count + 1);
    for (let i = 0; i <= count; i++) out[i] = i;
    return out;
}

function packBoolBytes(bools: Uint8Array): Uint8Array {
    const out = new Uint8Array((bools.length + 7) >>> 3);
    for (let i = 0; i < bools.length; i++) if (bools[i] !== 0) out[i >>> 3] |= 1 << (i & 7);
    return out;
}

const utf8Enc = new TextEncoder();
const pbfString = (s: string): string =>
    s.charCodeAt(0) === 0xfeff && utf8Enc.encode(s).length >= 12 ? s.slice(1) : s;

class DictStringVector extends Vector<Uint32Array, string> {
    constructor(
        name: string,
        codes: Uint32Array,
        private readonly dictionary: readonly string[],
        sizeOrNull: number | BitVector,
    ) {
        super(name, codes, sizeOrNull);
    }

    protected getValueFromBuffer(index: number): string {
        return pbfString(this.dictionary[this.dataBuffer[index]] ?? "");
    }
}

function v2PropertyToVector(col: V2PropertyColumn, numFeatures: number): Vector {
    const nullability = col.present ? new BitVector(col.present, numFeatures) : undefined;
    const sizeOrNull = nullability ?? numFeatures;
    if ("codes" in col) {
        return new DictStringVector(col.name, col.codes, col.dictionary, sizeOrNull);
    }
    switch (col.kind) {
        case "i32":
        case "u32":
            return new Int32FlatVector(col.name, col.values, sizeOrNull);
        case "i8":
        case "u8":
            return new Int32FlatVector(col.name, Int32Array.from(col.values), sizeOrNull);
        case "i64":
        case "u64":
            return new Int64FlatVector(col.name, col.values, sizeOrNull);
        case "f32":
            return new FloatFlatVector(col.name, col.values, sizeOrNull);
        case "f64":
            return new DoubleFlatVector(col.name, col.values, sizeOrNull);
        case "bool":
            return new BooleanFlatVector(col.name, new BitVector(packBoolBytes(col.values), numFeatures), sizeOrNull);
        case "str":
            return new ObjectFlatVector(col.name, col.values.map(pbfString), nullability);
    }
}

function v2LayerToFeatureTable(
    layer: V2Layer,
    geometryScaling: GeometryScaling | undefined,
    idWithinMaxSafeInteger: boolean,
): FeatureTable {
    const numFeatures = layer.featureCount;
    const geom = layer.geometry;
    let vertices = geom.vertices;

    if (geometryScaling) {
        const scale = geometryScaling.extent / layer.extent;
        geometryScaling.scale = scale;
        const { min, max } = geometryScaling;
        const scaled = new Int32Array(vertices.length);
        for (let i = 0; i < vertices.length; i += 2) {
            scaled[i] = Math.min(Math.max(Math.round(vertices[i] * scale), min), max);
            scaled[i + 1] = Math.min(Math.max(Math.round(vertices[i + 1] * scale), min), max);
        }
        vertices = scaled;
    }

    const types32 = Uint32Array.from(geom.types);
    let isConst = numFeatures > 0;
    let needsGeomOffsets = false;
    let needsPartOffsets = geom.ringOffsets !== undefined;

    for (let i = 0; i < numFeatures; i++) {
        const t = types32[i];
        if (t !== types32[0]) isConst = false;
        if (t === GEOMETRY_TYPE.MULTIPOINT || t === GEOMETRY_TYPE.MULTILINESTRING || t === GEOMETRY_TYPE.MULTIPOLYGON) {
            needsGeomOffsets = true;
        }
        if (
            t === GEOMETRY_TYPE.LINESTRING ||
            t === GEOMETRY_TYPE.MULTILINESTRING ||
            t === GEOMETRY_TYPE.POLYGON ||
            t === GEOMETRY_TYPE.MULTIPOLYGON
        ) {
            needsPartOffsets = true;
        }
    }

    let topology: TopologyVector | undefined;
    if (geom.layout !== "tessPolygons") {
        const geometryOffsets = geom.geometryOffsets ?? (needsGeomOffsets ? identityOffsets(numFeatures) : undefined);
        const numGeoms = geometryOffsets ? geometryOffsets[numFeatures] : numFeatures;
        topology = {
            geometryOffsets,
            partOffsets: geom.partOffsets ?? (needsPartOffsets ? identityOffsets(numGeoms) : undefined),
            ringOffsets: geom.ringOffsets,
        };
    }

    const geometryVector: GeometryVector | GpuVector =
        geom.indexBuffer && geom.triangleOffsets
            ? isConst
                ? createConstGpuVector(
                      numFeatures,
                      types32[0],
                      geom.triangleOffsets,
                      geom.indexBuffer,
                      vertices,
                      topology,
                  )
                : createFlatGpuVector(types32, geom.triangleOffsets, geom.indexBuffer, vertices, topology)
            : isConst
              ? new ConstGeometryVector(
                    numFeatures,
                    types32[0],
                    VertexBufferType.VEC_2,
                    topology ?? {},
                    undefined,
                    vertices,
                )
              : new FlatGeometryVector(VertexBufferType.VEC_2, types32, topology ?? {}, undefined, vertices);

    let idVector: IdVector | undefined;
    if (layer.ids) {
        const nullability = layer.ids.present ? new BitVector(layer.ids.present, numFeatures) : numFeatures;
        idVector =
            layer.ids.kind === "u32"
                ? new Int32FlatVector("_id", layer.ids.values, nullability)
                : idWithinMaxSafeInteger
                  ? new DoubleFlatVector("_id", Float64Array.from(layer.ids.values, Number), nullability)
                  : new Int64FlatVector("_id", layer.ids.values, nullability);
    }

    const propertyVectors: Vector[] = layer.properties.map((col) => v2PropertyToVector(col, numFeatures));
    return new FeatureTable(layer.name, geometryVector, idVector, propertyVectors, layer.extent, {
        version: 2,
        columnGeometry: geom,
    });
}

function decodeIdColumn(
    tile: Uint8Array,
    columnMetadata: Column,
    offset: IntWrapper,
    columnName: string,
    idDataStreamMetadata: StreamMetadata,
    sizeOrNullabilityBuffer: number | BitVector,
    idWithinMaxSafeInteger = false,
): IdVector {
    const idDataType = columnMetadata.scalarType?.longID ? ScalarType.UINT_64 : ScalarType.UINT_32;
    const nullabilityBuffer = typeof sizeOrNullabilityBuffer === "number" ? undefined : sizeOrNullabilityBuffer;

    const vectorType = getVectorType(
        idDataStreamMetadata,
        sizeOrNullabilityBuffer,
        tile,
        offset,
        idDataType === ScalarType.UINT_64 ? "int64" : "int32",
    );
    if (idDataType === ScalarType.UINT_32) {
        switch (vectorType) {
            case VectorType.FLAT: {
                const id = decodeUnsignedInt32Stream(tile, offset, idDataStreamMetadata, undefined, nullabilityBuffer);
                return new Int32FlatVector(columnName, id, sizeOrNullabilityBuffer);
            }
            case VectorType.SEQUENCE: {
                const id = decodeSequenceInt32Stream(tile, offset, idDataStreamMetadata);
                return new Int32SequenceVector(
                    columnName,
                    id[0],
                    id[1],
                    (idDataStreamMetadata as RleEncodedStreamMetadata).numRleValues,
                    false,
                );
            }
            case VectorType.CONST: {
                const id = decodeUnsignedConstInt32Stream(tile, offset, idDataStreamMetadata);
                return new Int32ConstVector(columnName, id, sizeOrNullabilityBuffer, false);
            }
        }
    }
    switch (vectorType) {
        case VectorType.FLAT: {
            if (idWithinMaxSafeInteger) {
                const id = decodeUnsignedInt64AsFloat64Stream(tile, offset, idDataStreamMetadata, nullabilityBuffer);
                return new DoubleFlatVector(columnName, id, sizeOrNullabilityBuffer);
            }
            const id = decodeUnsignedInt64Stream(tile, offset, idDataStreamMetadata, nullabilityBuffer);
            return new Int64FlatVector(columnName, id, sizeOrNullabilityBuffer);
        }
        case VectorType.SEQUENCE: {
            const id = decodeSequenceInt64Stream(tile, offset, idDataStreamMetadata);
            return new Int64SequenceVector(
                columnName,
                id[0],
                id[1],
                (idDataStreamMetadata as RleEncodedStreamMetadata).numRleValues,
                false,
            );
        }
        case VectorType.CONST: {
            const id = decodeUnsignedConstInt64Stream(tile, offset, idDataStreamMetadata);
            return new Int64ConstVector(columnName, id, sizeOrNullabilityBuffer, false);
        }
    }

    throw new Error("Vector type not supported for id column.");
}
