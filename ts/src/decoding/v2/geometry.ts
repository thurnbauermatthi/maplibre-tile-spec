import type { HeightMode, MltGeometryColumns, MltGeometryLayout } from "../../columns";
import {
    decodeLevel1LengthStream,
    decodeLevel1WithoutRingBufferLengthStream,
    decodeLevel2LengthStream,
    decodeRootLengthStream,
} from "../geometryDecoder";
import { EMPTY_I32, EMPTY_U32, allocI32, allocU8, allocU32, chargeBytes, checkScratchBytes } from "./binary";
import { decodePredictedVertices } from "./predicted";
import {
    LG_PREDICTED,
    NO_Z,
    type ParsedStream,
    ST_DATA_VERTEX,
    ST_LEN_GEOM,
    ST_LEN_PARTS,
    ST_LEN_RINGS,
    ST_LEN_TRIS,
    ST_OFF_TREE_CYC,
    ST_OFF_TREE_SYM,
    ST_OFF_VERTEX,
    decodeStreamI32,
    decodeStreamU32,
} from "./stream";

export const GEOM_POINT = 0;
export const GEOM_LINESTRING = 1;
export const GEOM_POLYGON = 2;
export const GEOM_MULTIPOINT = 3;
export const GEOM_MULTILINESTRING = 4;
export const GEOM_MULTIPOLYGON = 5;
export const GEOM_MESH = 6;

const isPolygon = (t: number): boolean => t === GEOM_POLYGON || t === GEOM_MULTIPOLYGON;
const isLine = (t: number): boolean => t === GEOM_LINESTRING || t === GEOM_MULTILINESTRING;

export interface TopologyLevels {
    geometryOffsets?: Uint32Array;
    partOffsets?: Uint32Array;
    ringOffsets?: Uint32Array;
}

const MAX_RETAINED_TREE_WORDS = 262144;
let sharedTreeSizes: Int32Array = new Int32Array(1024);
let sharedTreeFrames: Int32Array = new Int32Array(4096);

function decodeTrees(symbols: Uint32Array, cycles: Uint32Array): Uint32Array {
    const triangles = symbols.length;
    if (triangles === 0) {
        if (cycles.length !== 0) throw new Error("cycles");
        return EMPTY_U32;
    }
    if (symbols[triangles - 1] !== 0) throw new Error("tree symbols");

    const sizeWords = triangles + 1;
    const frameWords = (triangles + 2) << 2;
    checkScratchBytes((sizeWords + frameWords) * 4);
    let sizes: Int32Array;
    if (sizeWords <= sharedTreeSizes.length) {
        sizes = sharedTreeSizes;
        sizes[triangles] = 0;
    } else {
        sizes = new Int32Array(sizeWords);
        if (sizeWords <= MAX_RETAINED_TREE_WORDS) sharedTreeSizes = sizes;
    }
    let frames: Int32Array;
    if (frameWords <= sharedTreeFrames.length) {
        frames = sharedTreeFrames;
    } else {
        frames = new Int32Array(frameWords);
        if (frameWords <= MAX_RETAINED_TREE_WORDS) sharedTreeFrames = frames;
    }
    sizes[triangles - 1] = 4;

    let prevSz = 1;
    let bad = 0;
    for (let node = triangles - 2; node >= 0; node--) {
        const sym = symbols[node];
        bad |= sym >>> 2;
        const hasLeft = sym & 1;
        const hasRight = (sym >>> 1) & 1;
        const leftSize = prevSz & -hasLeft;
        const rightEntry = sizes[node + 1 + leftSize] >>> 2;
        bad |= hasRight & ((rightEntry - 1) >>> 31);
        prevSz = 1 + leftSize + (rightEntry & -hasRight);
        sizes[node] = (prevSz << 2) | sym;
    }
    if (bad !== 0) throw new Error("tree symbols");

    const out = allocU32(triangles * 3);
    let outPos = 0;
    let root = 0;
    let cyclePos = 0;
    while (root < triangles) {
        let cur = sizes[root];
        const treeSize = cur >>> 2;
        const nextCyclePos = cyclePos + 2 + treeSize;
        if (nextCyclePos > cycles.length) throw new Error("cycles eof");

        frames[0] = cycles[cyclePos];
        frames[1] = cycles[cyclePos + 1];
        frames[2] = 0;
        const apexBase = cyclePos + 2;
        const treeEnd = root + treeSize;
        let sp = 4;

        for (let node = root; node < treeEnd; node++) {
            sp -= 4;
            const from = frames[sp];
            const to = frames[sp + 1];
            const firstApex = frames[sp + 2];
            const next = sizes[node + 1];
            const sym = cur & 3;
            cur = next;
            const hasLeft = sym & 1;
            const hasRight = sym >>> 1;
            const leftSize = (next >>> 2) & -hasLeft;
            const apex = cycles[apexBase + firstApex + leftSize];

            out[outPos++] = from;
            out[outPos++] = to;
            out[outPos++] = apex;

            frames[sp] = from;
            frames[sp + 1] = apex;
            frames[sp + 2] = firstApex + leftSize + 1;
            sp += hasRight << 2;

            frames[sp] = apex;
            frames[sp + 1] = to;
            frames[sp + 2] = firstApex;
            sp += hasLeft << 2;
        }
        root = treeEnd;
        cyclePos = nextCyclePos;
    }
    if (cyclePos !== cycles.length) throw new Error("cycles trailing");
    return out;
}

export function decodeTopology(
    types: Uint8Array,
    geoLengths: Uint32Array | undefined,
    partLengths: Uint32Array | undefined,
    ringLengths: Uint32Array | undefined,
): TopologyLevels {
    let geometryOffsets = geoLengths;
    let partOffsets = partLengths;
    let ringOffsets = ringLengths;

    if (geometryOffsets !== undefined) {
        chargeBytes((types.length + 1) * 4);
        geometryOffsets = decodeRootLengthStream(types, geometryOffsets, GEOM_POLYGON);
        if (partOffsets !== undefined) {
            chargeBytes((geometryOffsets[geometryOffsets.length - 1] + 1) * 4);
            if (ringOffsets !== undefined) {
                partOffsets = decodeLevel1LengthStream(types, geometryOffsets, partOffsets, false);
                chargeBytes((partOffsets[partOffsets.length - 1] + 1) * 4);
                ringOffsets = decodeLevel2LengthStream(types, geometryOffsets, partOffsets, ringOffsets);
            } else {
                partOffsets = decodeLevel1WithoutRingBufferLengthStream(types, geometryOffsets, partOffsets);
            }
        }
    } else if (partOffsets !== undefined) {
        chargeBytes((types.length + 1) * 4);
        if (ringOffsets !== undefined) {
            let hasLine = false;
            for (let i = 0; i < types.length; i++) if (isLine(types[i])) hasLine = true;
            partOffsets = decodeRootLengthStream(types, partOffsets, GEOM_LINESTRING);
            chargeBytes((partOffsets[partOffsets.length - 1] + 1) * 4);
            ringOffsets = decodeLevel1LengthStream(types, partOffsets, ringOffsets, hasLine);
        } else {
            partOffsets = decodeRootLengthStream(types, partOffsets, GEOM_POINT);
        }
    }
    return { geometryOffsets, partOffsets, ringOffsets };
}

export function vertexIndexAt(levels: TopologyLevels, f: number): number {
    let idx = levels.geometryOffsets ? levels.geometryOffsets[f] : f;
    if (levels.partOffsets) idx = levels.partOffsets[idx];
    if (levels.ringOffsets) idx = levels.ringOffsets[idx];
    return idx;
}

export function decodeGeometry02(
    uniformType: number,
    featureCount: number,
    typesStream: ParsedStream | null,
    items: readonly ParsedStream[],
): MltGeometryColumns {
    const isMeshes = uniformType === GEOM_MESH;
    let types: Uint8Array;
    if (uniformType >= 0) {
        types = allocU8(featureCount).fill(uniformType);
    } else {
        if (!typesStream) throw new Error("missing types");
        const rawTypes = decodeStreamU32(typesStream, true);
        if (rawTypes.length !== featureCount) throw new Error("geom type count");
        types = allocU8(featureCount);
        for (let i = 0; i < featureCount; i++) {
            if (rawTypes[i] > GEOM_MULTIPOLYGON) throw new Error("geom type");
            types[i] = rawTypes[i];
        }
    }

    let geoLengths: Uint32Array | undefined;
    let partLengths: Uint32Array | undefined;
    let ringLengths: Uint32Array | undefined;
    let triLengths: Uint32Array | undefined;
    let vertexOffsets: Uint32Array | undefined;
    let treeSymbols: Uint32Array | undefined;
    let treeCycles: Uint32Array | undefined;
    let vertexStream: ParsedStream | null = null;

    for (const s of items) {
        if (s.t === ST_DATA_VERTEX) vertexStream = s;
        else if (s.t === ST_OFF_VERTEX) vertexOffsets = decodeStreamU32(s, true);
        else if (s.t === ST_OFF_TREE_SYM) treeSymbols = decodeStreamU32(s, true);
        else if (s.t === ST_OFF_TREE_CYC) treeCycles = decodeStreamU32(s, true);
        else if (s.t === ST_LEN_GEOM) geoLengths = decodeStreamU32(s, true);
        else if (s.t === ST_LEN_PARTS) partLengths = decodeStreamU32(s, true);
        else if (s.t === ST_LEN_RINGS) ringLengths = decodeStreamU32(s, true);
        else if (s.t === ST_LEN_TRIS) triLengths = decodeStreamU32(s, true);
        else throw new Error("geom st");
    }

    const zStep = vertexStream?.zStep ?? NO_Z;
    const heightMode = vertexStream?.heightMode ?? 0;
    if (isMeshes && (zStep === NO_Z || (geoLengths?.length ?? 0) !== types.length)) {
        throw new Error("mesh geom");
    }

    const indexBuffer = treeSymbols && treeCycles ? decodeTrees(treeSymbols, treeCycles) : undefined;
    if (!isMeshes && indexBuffer !== undefined && partLengths === undefined) {
        for (let i = 0; i < types.length; i++) if (!isPolygon(types[i])) throw new Error("tess non-poly");
    }

    const levels = decodeTopology(types, geoLengths, partLengths, ringLengths);
    const dimension: 2 | 3 = zStep !== NO_Z ? 3 : 2;
    const hasDict = vertexOffsets !== undefined;

    let vertices: Int32Array;
    if (!vertexStream) {
        vertices = EMPTY_I32;
    } else if (vertexStream.l === LG_PREDICTED) {
        if (hasDict) throw new Error("predicted dict");
        vertices = decodePredictedVertices(vertexStream.d, types, levels, vertexStream.n);
    } else {
        vertices = decodeStreamI32(vertexStream, hasDict);
    }

    let triangleOffsets: Uint32Array | undefined;
    if (triLengths !== undefined) {
        triangleOffsets = allocU32(triLengths.length + 1);
        let total = 0;
        for (let i = 0; i < triLengths.length; i++) {
            total += triLengths[i];
            if (total > 0xffffffff) throw new RangeError("overflow");
            triangleOffsets[i + 1] = total;
        }
    }

    if (treeCycles && indexBuffer) {
        if ((triangleOffsets?.[triangleOffsets.length - 1] ?? 0) !== indexBuffer.length / 3) {
            throw new Error("tri count");
        }
        const rawVertCount = vertices.length / dimension;
        for (let i = 0; i < treeCycles.length; i++) {
            if (treeCycles[i] >= rawVertCount) throw new RangeError("tcyc vtx");
        }
    }

    if (isMeshes) {
        const runs = levels.geometryOffsets ?? new Uint32Array([0]);
        const triOff = triangleOffsets ?? new Uint32Array([0]);
        const indices = indexBuffer ?? EMPTY_U32;
        if ((runs[runs.length - 1] ?? 0) !== vertices.length / 3 || triOff.length !== runs.length) {
            throw new Error("mesh len");
        }
        for (let f = 0; f < types.length; f++) {
            const v0 = runs[f];
            const v1 = runs[f + 1];
            const lo = triOff[f] * 3;
            const hi = triOff[f + 1] * 3;
            if (hi > indices.length) throw new Error("mesh idx");
            for (let k = lo; k < hi; k++) {
                if (indices[k] < v0 || indices[k] >= v1) throw new RangeError("mesh oob");
            }
        }
    }

    if (vertexOffsets !== undefined) {
        const dictCount = vertices.length / dimension;
        const expanded = allocI32(vertexOffsets.length * dimension);
        for (let i = 0; i < vertexOffsets.length; i++) {
            const idx = vertexOffsets[i];
            if (idx >= dictCount) throw new RangeError("vdict");
            for (let c = 0; c < dimension; c++) expanded[i * dimension + c] = vertices[idx * dimension + c];
        }
        vertices = expanded;
    }

    const layout: MltGeometryLayout = isMeshes
        ? "meshes"
        : indexBuffer !== undefined
          ? levels.partOffsets === undefined
              ? "tessPolygons"
              : "tessPolygonsWithOutlines"
          : "outlines";

    if (layout === "tessPolygons") {
        if ((triangleOffsets?.length ?? 0) !== types.length + 1) throw new Error("tri len");
    } else if (!isMeshes) {
        const { geometryOffsets, partOffsets, ringOffsets } = levels;
        const leaf = ringOffsets ?? partOffsets ?? geometryOffsets;
        const expectedVerts = leaf ? leaf[leaf.length - 1] : types.length;
        if (expectedVerts !== vertices.length / dimension) throw new Error("vtx count");
        let polyCount = 0;
        for (let f = 0; f < types.length; f++) {
            const t = types[f];
            const g0 = geometryOffsets ? geometryOffsets[f] : f;
            const g1 = geometryOffsets ? geometryOffsets[f + 1] : f + 1;
            if (isPolygon(t)) {
                if (g1 > g0 && (!partOffsets || !ringOffsets)) throw new Error("poly offsets");
                if (layout === "tessPolygonsWithOutlines" && triangleOffsets && indexBuffer) {
                    if (polyCount + 1 >= triangleOffsets.length) throw new Error("tri len");
                    const v0 = vertexIndexAt(levels, f);
                    const v1 = vertexIndexAt(levels, f + 1);
                    const lo = triangleOffsets[polyCount] * 3;
                    const hi = triangleOffsets[polyCount + 1] * 3;
                    for (let k = lo; k < hi; k++) {
                        if (indexBuffer[k] < v0 || indexBuffer[k] >= v1) throw new RangeError("tess oob");
                    }
                }
                polyCount++;
            } else if (isLine(t)) {
                if (g1 > g0 && !partOffsets) throw new Error("line offsets");
            }
        }
        if (layout === "tessPolygonsWithOutlines" && (triangleOffsets?.length ?? 0) !== polyCount + 1) {
            throw new Error("tri len");
        }
    }

    const hm: HeightMode = heightMode === 1 ? "relativeToGround" : "absolute";
    return {
        types,
        layout,
        ...(isMeshes ? { meshes: true as const } : {}),
        dimension,
        vertices,
        ...(levels.geometryOffsets ? { geometryOffsets: levels.geometryOffsets } : {}),
        ...(levels.partOffsets ? { partOffsets: levels.partOffsets } : {}),
        ...(levels.ringOffsets ? { ringOffsets: levels.ringOffsets } : {}),
        ...(triangleOffsets ? { triangleOffsets } : {}),
        ...(indexBuffer ? { indexBuffer } : {}),
        ...(zStep !== NO_Z ? { zStep, heightMode: hm } : {}),
    };
}
