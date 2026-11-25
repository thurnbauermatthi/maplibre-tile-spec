import { decodeBigEndianInt32sInto } from "../bigEndianDecode";
import { createFastPforWireDecodeWorkspace, decodeFastPforInt32 } from "../fastPforDecoder";
import {
    type ByteReader,
    allocF32,
    allocF64,
    allocI8,
    allocI64,
    allocU8,
    allocU32,
    allocU64,
    asI32,
    asI64,
    checkScratchBytes,
    copyLittleEndian,
    cwDeltaVecInPlace,
    decodeMortonDelta,
    decodeRleInterleavedU32,
    decodeRleInterleavedU64,
    decodeVarintU32Into,
    decodeVarintU32sAll,
    decodeVarintU64Into,
    decodeVarintU64sAll,
    readBytes,
    readU8,
    readVarintU32,
    scratchU32,
    scratchU64,
    zigzagDelta32InPlace,
    zigzagDelta64InPlace,
    zigzagI32InPlace,
    zigzagI64InPlace,
} from "./binary";
import { PRES_BITMAP, PRES_BYTE_RLE, readPresenceBits } from "./presence";

export const ST_PRESENT = 0;
export const ST_DATA_NONE = 1;
export const ST_DATA_SINGLE = 2;
export const ST_DATA_SHARED = 3;
export const ST_DATA_VERTEX = 4;
export const ST_DATA_FSST = 6;
export const ST_OFF_VERTEX = 7;
export const ST_OFF_STRING = 9;
export const ST_OFF_KEY = 10;
export const ST_OFF_TREE_SYM = 11;
export const ST_OFF_TREE_CYC = 12;
export const ST_LEN_VARBIN = 13;
export const ST_LEN_GEOM = 14;
export const ST_LEN_PARTS = 15;
export const ST_LEN_RINGS = 16;
export const ST_LEN_TRIS = 17;
export const ST_LEN_SYM = 18;
export const ST_LEN_DICT = 19;
export const ST_LEN_NESTED = 20;

export const PH_NONE = 0;
export const PH_FPF256 = 2;
export const PH_VARINT = 3;

export const LG_NONE = 0;
export const LG_DELTA = 1;
export const LG_RLE = 3;
export const LG_DELTA_RLE = 4;
export const LG_CW_DELTA = 5;
export const LG_MORTON = 7;
export const LG_PREDICTED = 8;
export const LG_BYTE_RLE = 10;

export const FM_INT = 0;
export const FM_STR = 1;
export const FM_BOOL = 2;
export const FM_FLOAT = 3;
export const FM_VERTEX = 4;
export const FM_BYTES = 5;

export const HAS_EXPLICIT_COUNT = 0x80;
export const NO_Z = 99;

const MAX_RETAINED_FPF_WORDS = 262144;
const sharedFpfWorkspace = createFastPforWireDecodeWorkspace(256);

const FAMILY_TABLE: readonly (readonly number[])[] = [
    [LG_NONE, LG_DELTA, LG_RLE, LG_DELTA_RLE],
    [LG_NONE, LG_DELTA, LG_RLE, LG_DELTA_RLE],
    [LG_NONE, LG_BYTE_RLE],
    [LG_NONE],
    [LG_NONE, LG_DELTA, LG_CW_DELTA, LG_MORTON, LG_PREDICTED],
    [LG_NONE, 200],
];

export interface ParsedStream {
    t: number;
    l: number;
    p: number;
    n: number;
    w: number;
    mBits: number;
    mShift: number;
    zStep: number;
    heightMode: number;
    front: number;
    d: Uint8Array;
}

export function parseStream02(
    r: ByteReader,
    family: number,
    wordWidth: number,
    t: number,
    impliedCount: number,
    expectedStrLayout = -1,
): ParsedStream {
    const encByte = readU8(r);
    const explicit = (encByte & HAS_EXPLICIT_COUNT) !== 0;
    const code = (encByte & 0x70) >>> 4;
    const physField = encByte & 0x0c;
    const extField = encByte & 0x03;

    const members = FAMILY_TABLE[family];
    if (code >= members.length || members[code] < 0) throw new Error("v2 enc");
    let l = members[code];

    if (family === FM_STR) {
        if (expectedStrLayout >= 0 && extField !== expectedStrLayout) throw new Error("v2 str ext");
    } else if (family === FM_VERTEX) {
        if (extField > 1) throw new Error("v2 vtx ext");
    } else if (extField !== 0) {
        throw new Error("v2 ext");
    }

    const isXyz = family === FM_VERTEX && extField === 1;
    let p = PH_NONE;
    let omitsLen = false;
    let front = 0;

    const physInt = (): number => {
        if (physField === 0x04) return PH_NONE;
        if (physField === 0x08) return PH_VARINT;
        if (physField === 0x0c) return PH_FPF256;
        throw new Error("v2 phys");
    };
    const rawInt = (): number => {
        if (physField === 0x00) {
            omitsLen = true;
            return PH_NONE;
        }
        return physInt();
    };

    if (family === FM_INT || family === FM_STR) {
        if (l === LG_NONE) p = rawInt();
        else if (l === LG_DELTA) p = physInt();
        else {
            if (physField !== 0) throw new Error("v2 rle phys");
            p = PH_VARINT;
        }
    } else if (family === FM_BYTES) {
        if (explicit || physField !== 0x04) throw new Error("v2 bytes");
        if (l === 200) {
            if (t !== ST_DATA_SINGLE && t !== ST_DATA_SHARED) throw new Error("v2 front");
            front = 1;
            l = LG_NONE;
        }
    } else if (family === FM_BOOL) {
        if (physField !== 0) throw new Error("v2 bool");
    } else if (family === FM_FLOAT) {
        if (physField === 0x00) omitsLen = true;
        else if (physField !== 0x04) throw new Error("v2 float");
    } else if (isXyz) {
        if (l === LG_NONE) p = rawInt();
        else if (l === LG_DELTA || l === LG_CW_DELTA) p = physInt();
        else throw new Error("v2 xyz");
    } else {
        if (l === LG_NONE) p = rawInt();
        else if (l === LG_PREDICTED) {
            if (physField !== 0) throw new Error("v2 predicted phys");
        } else p = physInt();
    }

    if (family === FM_BYTES) {
        const byteLen = readVarintU32(r);
        return {
            t,
            l,
            p,
            n: byteLen,
            w: byteLen,
            mBits: 0,
            mShift: 0,
            zStep: NO_Z,
            heightMode: 0,
            front,
            d: readBytes(r, byteLen),
        };
    }

    const n = explicit ? readVarintU32(r) : impliedCount;
    if (n < 0) throw new Error("v2 no count");
    const wpv = isXyz ? 3 : family === FM_VERTEX && l !== LG_MORTON ? 2 : 1;
    const w = n * wpv;
    if (!Number.isSafeInteger(w) || w > 0xffffffff) throw new RangeError("overflow");

    if (family === FM_BOOL) {
        const coding = l === LG_NONE ? PRES_BITMAP : PRES_BYTE_RLE;
        const d = readPresenceBits(r, n, coding);
        return {
            t,
            l: LG_NONE,
            p: PH_NONE,
            n,
            w,
            mBits: 0,
            mShift: 0,
            zStep: NO_Z,
            heightMode: 0,
            front: 0,
            d,
        };
    }

    const byteLen = omitsLen ? w * wordWidth : readVarintU32(r);
    if (!Number.isSafeInteger(byteLen) || byteLen > 0xffffffff) throw new RangeError("overflow");

    let mBits = 0;
    let mShift = 0;
    let zStep = NO_Z;
    let heightMode = 0;

    if (isXyz) {
        const cb = readU8(r);
        if (cb >= 16) throw new Error("z_step");
        zStep = (cb & 7) - 3;
        heightMode = (cb & 8) >>> 3;
    } else if (family === FM_VERTEX && l === LG_MORTON) {
        mBits = readVarintU32(r);
        mShift = readVarintU32(r);
        if (mBits > 16) throw new Error("morton bits");
    }

    return { t, l, p, n, w, mBits, mShift, zStep, heightMode, front, d: readBytes(r, byteLen) };
}

export function decodeBoolValues(s: ParsedStream): Uint8Array {
    const out = allocU8(s.n);
    for (let i = 0; i < s.n; i++) out[i] = (s.d[i >>> 3] >>> (i & 7)) & 1;
    return out;
}

function decodeFastPfor256Into(data: Uint8Array, numValues: number, out: Uint32Array): Uint32Array {
    if (numValues === 0) {
        if (data.length !== 0) throw new Error("fpf empty");
        return out;
    }
    if ((data.length & 3) !== 0) throw new Error("fpf byte len");
    checkScratchBytes(data.length);
    const wordCount = data.length >>> 2;
    let words = sharedFpfWorkspace.encodedWords;
    if (words.length < wordCount) {
        words = new Uint32Array(Math.max(256, wordCount));
        if (words.length <= MAX_RETAINED_FPF_WORDS) sharedFpfWorkspace.encodedWords = words;
    }
    decodeBigEndianInt32sInto(data, 0, data.length, words);
    return decodeFastPforInt32(words.subarray(0, wordCount), numValues, sharedFpfWorkspace.decoderWorkspace, out);
}

function decodePhysicalU32(s: ParsedStream, out: Uint32Array): Uint32Array {
    if (s.p === PH_NONE) return copyLittleEndian(Uint32Array, s.d, s.w, out);
    if (s.p === PH_FPF256) return decodeFastPfor256Into(s.d, s.w, out);
    if (s.p === PH_VARINT) return decodeVarintU32Into(s.d, s.w, out);
    throw new Error("u32 phys");
}

function decodePhysicalU64(s: ParsedStream): BigUint64Array {
    if ((s.l === LG_RLE || s.l === LG_DELTA_RLE) && s.p === PH_VARINT) return decodeVarintU64sAll(s.d);
    const out = scratchU64(s.w);
    if (s.p === PH_NONE) return copyLittleEndian(BigUint64Array, s.d, s.w, out);
    if (s.p === PH_VARINT) return decodeVarintU64Into(s.d, s.w, out);
    throw new Error("u64 phys");
}

function decodeRleU32(s: ParsedStream, scratch: boolean): Uint32Array {
    const words = s.p === PH_VARINT ? decodeVarintU32sAll(s.d) : decodePhysicalU32(s, scratchU32(s.w));
    return decodeRleInterleavedU32(words, s.n, scratch ? scratchU32(s.n) : allocU32(s.n));
}

export function decodeStreamU32(s: ParsedStream, scratch = false): Uint32Array {
    if (s.l === LG_RLE) return decodeRleU32(s, scratch);
    if (s.l === LG_DELTA_RLE) return zigzagDelta32InPlace(decodeRleU32(s, scratch));
    const out = decodePhysicalU32(s, scratch ? scratchU32(s.w) : allocU32(s.w));
    if (s.l === LG_NONE) return out;
    if (s.l === LG_DELTA) return zigzagDelta32InPlace(out);
    throw new Error("u32 log");
}

export function decodeStreamI32(s: ParsedStream, scratch = false): Int32Array {
    if (s.l === LG_MORTON) return decodeMortonDelta(decodePhysicalU32(s, scratchU32(s.w)), s.mBits, s.mShift);
    if (s.l === LG_RLE) return zigzagI32InPlace(decodeRleU32(s, scratch));
    if (s.l === LG_DELTA_RLE) return asI32(zigzagDelta32InPlace(decodeRleU32(s, scratch)));

    const out = decodePhysicalU32(s, scratch ? scratchU32(s.w) : allocU32(s.w));
    if (s.zStep !== NO_Z) {
        if (s.l === LG_NONE) return zigzagI32InPlace(out);
        if (s.l === LG_DELTA) return asI32(zigzagDelta32InPlace(out));
        if (s.l === LG_CW_DELTA) return cwDeltaVecInPlace(out, 3);
        throw new Error("xyz log");
    }
    if (s.l === LG_NONE) return zigzagI32InPlace(out);
    if (s.l === LG_DELTA) return asI32(zigzagDelta32InPlace(out));
    if (s.l === LG_CW_DELTA) return cwDeltaVecInPlace(out, 2);
    throw new Error("i32 log");
}

export function decodeStreamI8(s: ParsedStream): Int8Array {
    const i32s = decodeStreamI32(s, true);
    const out = allocI8(i32s.length);
    for (let i = 0; i < i32s.length; i++) {
        if (i32s[i] < -128 || i32s[i] > 127) throw new RangeError("i8");
        out[i] = i32s[i];
    }
    return out;
}

export function decodeStreamU8(s: ParsedStream): Uint8Array {
    const u32s = decodeStreamU32(s, true);
    const out = allocU8(u32s.length);
    for (let i = 0; i < u32s.length; i++) {
        if (u32s[i] > 255) throw new RangeError("u8");
        out[i] = u32s[i];
    }
    return out;
}

function decodeWords64(s: ParsedStream, owned: boolean, signed: boolean): BigUint64Array {
    if (s.l === LG_RLE || s.l === LG_DELTA_RLE) {
        const buf = owned ? (signed ? new BigUint64Array(allocI64(s.n).buffer) : allocU64(s.n)) : scratchU64(s.n);
        const words = decodeRleInterleavedU64(decodePhysicalU64(s), s.n, buf);
        return s.l === LG_DELTA_RLE ? zigzagDelta64InPlace(words) : words;
    }
    const phys = decodePhysicalU64(s);
    if (!owned) return phys;
    const words = signed ? new BigUint64Array(allocI64(s.w).buffer) : allocU64(s.w);
    words.set(phys);
    return words;
}

export function decodeStreamI64(s: ParsedStream, owned = false): BigInt64Array {
    const words = decodeWords64(s, owned, true);
    if (s.l === LG_RLE || s.l === LG_NONE) return zigzagI64InPlace(words);
    if (s.l === LG_DELTA_RLE) return asI64(words);
    if (s.l === LG_DELTA) return asI64(zigzagDelta64InPlace(words));
    throw new Error("64 log");
}

export function decodeStreamU64(s: ParsedStream, owned = false): BigUint64Array {
    const words = decodeWords64(s, owned, false);
    if (s.l === LG_RLE || s.l === LG_DELTA_RLE || s.l === LG_NONE) return words;
    if (s.l === LG_DELTA) return zigzagDelta64InPlace(words);
    throw new Error("64 log");
}

export function decodeFloatValues(dataStream: ParsedStream, isF32: boolean): Float32Array | Float64Array {
    if (dataStream.l !== LG_NONE || dataStream.p !== PH_NONE) throw new Error("float enc");
    return isF32
        ? copyLittleEndian(Float32Array, dataStream.d, dataStream.n, allocF32(dataStream.n))
        : copyLittleEndian(Float64Array, dataStream.d, dataStream.n, allocF64(dataStream.n));
}
