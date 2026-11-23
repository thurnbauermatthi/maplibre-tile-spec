import {
    type ByteReader,
    allocF32,
    allocF64,
    allocI8,
    allocI32,
    allocI64,
    allocU8,
    allocU32,
    allocU64,
    cursor,
    peekU8,
    readStr,
    readU8,
    readVarintU32,
    resetBudget,
} from "./binary";
import { decodeGeometry02 } from "./geometry";
import {
    type MutableIndexable,
    PRES_BITMAP,
    bitmapPopcount,
    normalizePresentBitmap,
    readPresenceBits,
    scatterPresent,
} from "./presence";
import {
    FM_BOOL,
    FM_BYTES,
    FM_FLOAT,
    FM_INT,
    FM_STR,
    FM_VERTEX,
    HAS_EXPLICIT_COUNT,
    LG_DICT,
    type ParsedStream,
    ST_DATA_FSST,
    ST_DATA_NONE,
    ST_DATA_SHARED,
    ST_DATA_SINGLE,
    ST_DATA_VERTEX,
    ST_LEN_DICT,
    ST_LEN_GEOM,
    ST_LEN_PARTS,
    ST_LEN_RINGS,
    ST_LEN_SYM,
    ST_LEN_TRIS,
    ST_LEN_VARBIN,
    ST_OFF_STRING,
    ST_OFF_TREE_CYC,
    ST_OFF_TREE_SYM,
    ST_OFF_VERTEX,
    decodeBoolValues,
    decodeFloatValues,
    decodeStreamI8,
    decodeStreamI32,
    decodeStreamI64,
    decodeStreamU8,
    decodeStreamU32,
    decodeStreamU64,
    parseStream02,
} from "./stream";
import {
    type RawSharedDict,
    type RawStringsStreamSet,
    decodeSharedDictColumns,
    decodeStringsDense,
    materializeStringProperty,
    parseStrings02,
} from "./strings";

export type V2GeometryLayout = "outlines" | "tessPolygons" | "tessPolygonsWithOutlines";

export interface V2GeometryColumns {
    readonly types: Uint8Array;
    readonly layout: V2GeometryLayout;
    readonly dimension: 2;
    readonly vertices: Int32Array;
    readonly geometryOffsets?: Uint32Array;
    readonly partOffsets?: Uint32Array;
    readonly ringOffsets?: Uint32Array;
    readonly triangleOffsets?: Uint32Array;
    readonly indexBuffer?: Uint32Array;
}

export type V2IdColumn =
    | { readonly kind: "u32"; readonly present?: Uint8Array; readonly values: Uint32Array }
    | { readonly kind: "u64"; readonly present?: Uint8Array; readonly values: BigUint64Array };

export interface V2ScalarPropertyColumn<K extends string, A> {
    readonly name: string;
    readonly kind: K;
    readonly present?: Uint8Array;
    readonly values: A;
}

export type V2StringPropertyColumn =
    | { readonly name: string; readonly kind: "str"; readonly present?: Uint8Array; readonly values: readonly string[] }
    | {
          readonly name: string;
          readonly kind: "strDict";
          readonly present?: Uint8Array;
          readonly codes: Uint32Array;
          readonly dictionary: readonly string[];
      };

export type V2PropertyColumn =
    | V2ScalarPropertyColumn<"bool", Uint8Array>
    | V2ScalarPropertyColumn<"i8", Int8Array>
    | V2ScalarPropertyColumn<"u8", Uint8Array>
    | V2ScalarPropertyColumn<"i32", Int32Array>
    | V2ScalarPropertyColumn<"u32", Uint32Array>
    | V2ScalarPropertyColumn<"i64", BigInt64Array>
    | V2ScalarPropertyColumn<"u64", BigUint64Array>
    | V2ScalarPropertyColumn<"f32", Float32Array>
    | V2ScalarPropertyColumn<"f64", Float64Array>
    | V2StringPropertyColumn;

export interface V2Layer {
    readonly name: string;
    readonly extent: number;
    readonly version: 2;
    readonly featureCount: number;
    readonly geometry: V2GeometryColumns;
    readonly ids?: V2IdColumn;
    readonly properties: readonly V2PropertyColumn[];
}

type RawScalarColumn02 =
    | { dt: 2 | 3 | 4 | 5 | 6 | 7 | 8; s: ParsedStream; d: null }
    | { dt: 9 | 10; s: ParsedStream; d: ParsedStream | null };

type RawColumnValues02 = RawScalarColumn02 | { dt: 11; s: RawStringsStreamSet; d: null };

type DecodedScalar =
    | { kind: "bool"; values: Uint8Array }
    | { kind: "i8"; values: Int8Array }
    | { kind: "u8"; values: Uint8Array }
    | { kind: "i32"; values: Int32Array }
    | { kind: "u32"; values: Uint32Array }
    | { kind: "i64"; values: BigInt64Array }
    | { kind: "u64"; values: BigUint64Array }
    | { kind: "f32"; values: Float32Array }
    | { kind: "f64"; values: Float64Array };

function decodeScalarValues(cv: RawScalarColumn02, scratch = false, owned64 = false): DecodedScalar {
    switch (cv.dt) {
        case 2:
            return { kind: "bool", values: decodeBoolValues(cv.s) };
        case 3:
            return { kind: "i8", values: decodeStreamI8(cv.s) };
        case 4:
            return { kind: "u8", values: decodeStreamU8(cv.s) };
        case 5:
            return { kind: "i32", values: decodeStreamI32(cv.s, scratch) };
        case 6:
            return { kind: "u32", values: decodeStreamU32(cv.s, scratch) };
        case 7:
            return { kind: "i64", values: decodeStreamI64(cv.s, owned64) };
        case 8:
            return { kind: "u64", values: decodeStreamU64(cv.s, owned64) };
        case 9:
            return { kind: "f32", values: decodeFloatValues(cv.s, cv.d, true) as Float32Array };
        case 10:
            return { kind: "f64", values: decodeFloatValues(cv.s, cv.d, false) as Float64Array };
    }
}

function parseColumnValues02(r: ByteReader, dt: number, impliedCount: number): RawColumnValues02 {
    if (impliedCount < 0 && (peekU8(r) & HAS_EXPLICIT_COUNT) === 0) throw new Error("explicit count");
    if (dt === 2) return { dt: 2, s: parseStream02(r, FM_BOOL, 4, ST_DATA_NONE, impliedCount), d: null };
    if (dt >= 3 && dt <= 6)
        return { dt: dt as 3 | 4 | 5 | 6, s: parseStream02(r, FM_INT, 4, ST_DATA_NONE, impliedCount), d: null };
    if (dt === 7 || dt === 8) return { dt, s: parseStream02(r, FM_INT, 8, ST_DATA_NONE, impliedCount), d: null };
    if (dt === 9 || dt === 10) {
        const w = dt === 9 ? 4 : 8;
        const s = parseStream02(r, FM_FLOAT, w, ST_DATA_NONE, impliedCount);
        return { dt, s, d: s.l === LG_DICT ? parseStream02(r, FM_FLOAT, w, ST_DATA_SINGLE, s.n) : null };
    }
    if (dt === 11) return { dt: 11, s: parseStrings02(r, impliedCount), d: null };
    throw new Error("col type");
}

type TypedCol<V> = ArrayLike<V> & MutableIndexable<V> & { set(src: ArrayLike<V>): void };

function expandCol<V extends number | bigint, A extends TypedCol<V>>(
    dense: A,
    bits: Uint8Array | undefined,
    featureCount: number,
    alloc: (n: number) => A,
    copyDense: boolean,
): { values: A; present?: Uint8Array } {
    if (bits === undefined) {
        if (dense.length !== featureCount) throw new Error("col count");
        if (!copyDense) return { values: dense };
        const out = alloc(featureCount);
        out.set(dense);
        return { values: out };
    }
    const pop = bitmapPopcount(bits, featureCount);
    if (pop !== dense.length) throw new Error("popcount");
    const present = normalizePresentBitmap(bits, featureCount);
    const values = alloc(featureCount);
    if (present === undefined) {
        values.set(dense);
        return { values };
    }
    if (pop > 0) scatterPresent(bits, featureCount, dense, values);
    return { values, present };
}

function decodeValueProp02(
    name: string,
    cv: RawColumnValues02,
    bits: Uint8Array | undefined,
    featureCount: number,
): V2PropertyColumn {
    if (cv.dt === 11) {
        const owned = bits === undefined;
        return materializeStringProperty(name, decodeStringsDense(cv.s, owned), bits, featureCount, owned);
    }
    const scratch = bits !== undefined;
    const dec = decodeScalarValues(cv, scratch, false);
    switch (dec.kind) {
        case "bool":
            return { name, kind: "bool", ...expandCol(dec.values, bits, featureCount, allocU8, false) };
        case "i8":
            return { name, kind: "i8", ...expandCol(dec.values, bits, featureCount, allocI8, false) };
        case "u8":
            return { name, kind: "u8", ...expandCol(dec.values, bits, featureCount, allocU8, false) };
        case "i32":
            return { name, kind: "i32", ...expandCol(dec.values, bits, featureCount, allocI32, scratch) };
        case "u32":
            return { name, kind: "u32", ...expandCol(dec.values, bits, featureCount, allocU32, scratch) };
        case "i64":
            return { name, kind: "i64", ...expandCol(dec.values, bits, featureCount, allocI64, true) };
        case "u64":
            return { name, kind: "u64", ...expandCol(dec.values, bits, featureCount, allocU64, true) };
        case "f32":
            return { name, kind: "f32", ...expandCol(dec.values, bits, featureCount, allocF32, false) };
        case "f64":
            return { name, kind: "f64", ...expandCol(dec.values, bits, featureCount, allocF64, false) };
    }
}

function readColPresence02(
    r: ByteReader,
    high: number,
    sharedBitfields: readonly Uint8Array[],
    featureCount: number,
): Uint8Array | undefined {
    if (high === 0x00) return undefined;
    if (high < 0x40) {
        const code = (high >>> 4) - 1;
        if (code > 2) throw new Error("v2 pres");
        return readPresenceBits(r, featureCount, code);
    }
    const idx = (high - 0x40) >>> 4;
    if (idx >= sharedBitfields.length) throw new Error("v2 spres");
    return sharedBitfields[idx];
}

const GEO_HAS_GEO = 0x6ccc;
const GEO_HAS_PARTS = 0x2ff0;
const GEO_HAS_RINGS = 0x2f00;

function decodeLayer02(r: ByteReader): V2Layer {
    const name = readStr(r);
    if (name.length === 0) throw new Error("v2 name");

    const hdr = readU8(r);
    const hasMValues = (hdr & 0x80) !== 0;
    if (hasMValues) throw new Error("v2 mvalues unsupported");
    const uniNib = (hdr & 0x70) >>> 4;
    if (uniNib > 6) throw new Error("v2 hdr");
    const uniformType = uniNib === 0 ? -1 : uniNib - 1;
    const extent = 1 << ((hdr & 0x0f) + 6);
    const featureCount = readVarintU32(r);

    const layout = readU8(r);
    const sharedCoded = (layout & 0x80) !== 0;
    const sharedCount = (layout & 0x70) >>> 4;
    const geoLayout = layout & 0x0f;
    if ((sharedCoded && sharedCount === 0) || geoLayout > 0x0d) throw new Error("v2 layout");

    const sharedBitfields: Uint8Array[] = new Array(sharedCount);
    for (let i = 0; i < sharedCount; i++) {
        let coding = PRES_BITMAP;
        if (sharedCoded) {
            const cb = readU8(r);
            if ((cb & 0x8f) !== 0) throw new Error("v2 spres byte");
            coding = (cb & 0x70) >>> 4;
            if (coding > 2) throw new Error("v2 spres code");
        }
        sharedBitfields[i] = readPresenceBits(r, featureCount, coding);
    }

    const typesStream = uniformType >= 0 ? null : parseStream02(r, FM_INT, 4, ST_LEN_VARBIN, featureCount);
    const items: ParsedStream[] = [];
    if (((GEO_HAS_GEO >>> geoLayout) & 1) !== 0) items.push(parseStream02(r, FM_INT, 4, ST_LEN_GEOM, featureCount));
    if (((GEO_HAS_PARTS >>> geoLayout) & 1) !== 0) items.push(parseStream02(r, FM_INT, 4, ST_LEN_PARTS, featureCount));
    if (((GEO_HAS_RINGS >>> geoLayout) & 1) !== 0) items.push(parseStream02(r, FM_INT, 4, ST_LEN_RINGS, featureCount));
    if (geoLayout >= 0x0c) {
        items.push(
            parseStream02(r, FM_INT, 4, ST_LEN_TRIS, featureCount),
            parseStream02(r, FM_INT, 4, ST_OFF_TREE_SYM, featureCount),
            parseStream02(r, FM_INT, 4, ST_OFF_TREE_CYC, featureCount),
        );
    }
    items.push(parseStream02(r, FM_VERTEX, 4, ST_DATA_VERTEX, featureCount));
    if (geoLayout <= 0x0b && (geoLayout & 1) === 1)
        items.push(parseStream02(r, FM_INT, 4, ST_OFF_VERTEX, featureCount));

    const geometry = decodeGeometry02(uniformType, featureCount, typesStream, items);

    const columnCount = readVarintU32(r);
    if (r.e - r.p < columnCount) throw new RangeError("v2 cols eof");

    let ids: V2IdColumn | undefined;
    const properties: V2PropertyColumn[] = [];
    const seen = new Set<string>();
    const reg = (n: string): void => {
        if (seen.has(n)) throw new Error("dup col");
        seen.add(n);
    };

    for (let i = 0; i < columnCount; i++) {
        const tb = readU8(r);
        const high = tb & 0xf0;
        const data = tb & 0x0f;

        if (data === 0x0f) {
            if (high !== 0x00 && high !== 0x10) throw new Error("v2 sdict");
            const prefix = readStr(r);
            const cc = readVarintU32(r);
            if (r.e - r.p < cc) throw new RangeError("v2 sdict eof");
            const lengths = parseStream02(r, FM_INT, 4, ST_LEN_DICT, -1);
            let shared: RawSharedDict;
            if (high === 0x00) {
                const blob = parseStream02(r, FM_BYTES, 4, ST_DATA_SHARED, -1);
                shared = { prefix, front: blob.front, lengths, data: blob, children: new Array(cc) };
            } else {
                const symLens = parseStream02(r, FM_INT, 4, ST_LEN_SYM, -1);
                const symTbl = parseStream02(r, FM_BYTES, 4, ST_DATA_FSST, -1);
                const corpus = parseStream02(r, FM_BYTES, 4, ST_DATA_SHARED, -1);
                shared = { prefix, front: corpus.front, lengths, symLens, symTbl, corpus, children: new Array(cc) };
            }
            for (let c = 0; c < cc; c++) {
                const cb = readU8(r);
                if ((cb & 0x0f) !== 0x0b) throw new Error("v2 sdict ch");
                const suffix = readStr(r);
                reg(`${prefix}${suffix}`);
                const cBits = readColPresence02(r, cb & 0xf0, sharedBitfields, featureCount);
                const cCount = cBits === undefined ? featureCount : bitmapPopcount(cBits, featureCount);
                shared.children[c] = {
                    suffix,
                    bits: cBits,
                    data: parseStream02(r, FM_STR, 4, ST_OFF_STRING, cCount, 1),
                };
            }
            properties.push(...decodeSharedDictColumns(shared, featureCount));
            continue;
        }

        const colName = data >= 2 ? readStr(r) : "";
        const bits = readColPresence02(r, high, sharedBitfields, featureCount);
        const dCount = bits === undefined ? featureCount : bitmapPopcount(bits, featureCount);

        if (data <= 1) {
            if (ids !== undefined) throw new Error("dup id");
            const s = parseStream02(r, FM_INT, data === 1 ? 8 : 4, ST_DATA_NONE, dCount);
            ids =
                data === 1
                    ? { kind: "u64", ...expandCol(decodeStreamU64(s, false), bits, featureCount, allocU64, true) }
                    : { kind: "u32", ...expandCol(decodeStreamU32(s, true), bits, featureCount, allocU32, true) };
        } else if (data <= 11) {
            reg(colName);
            properties.push(decodeValueProp02(colName, parseColumnValues02(r, data, dCount), bits, featureCount));
        } else {
            throw new Error("v2 nested unsupported");
        }
    }

    if (r.p !== r.e) throw new Error("v2 trailing");

    return {
        name,
        extent,
        version: 2,
        featureCount,
        geometry,
        ...(ids !== undefined ? { ids } : {}),
        properties,
    };
}

export function decodeSingleLayer02(bodyBytes: Uint8Array, maxBytes?: number, resetBudgetOnStart = true): V2Layer {
    if (resetBudgetOnStart) resetBudget(maxBytes);
    return decodeLayer02(cursor(bodyBytes));
}
