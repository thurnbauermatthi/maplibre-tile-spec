import { EMPTY_I32, allocI32, checkScratchBytes, cursor, readBytes, readVarintU32 } from "./binary";

const STATE_LOW = 65536;
const SYMBOLS = 257;
const INVALID_SYM = 257;
const TABLE_LEN = 258;
const CONTEXTS = 14;
const MAX_RETAINED_FIRST = 262144;

const lowMask = (b: number): number => (b === 0 ? 0 : 0xffffffff >>> (32 - b));

interface SymInfo {
    rawBits: Uint8Array;
    dyCtxBase: Uint8Array;
    valid: Uint8Array;
    headShifted: Int32Array;
    signMask: Int32Array;
}

let symInfo: SymInfo[] | null = null;
let sharedSlab: Uint32Array = new Uint32Array(4096);
let sharedFirst: Uint8Array = new Uint8Array(4096);
const sharedFreq = new Uint16Array(SYMBOLS);
const ctxStart = new Uint32Array(CONTEXTS);
const ctxPrecision = new Uint8Array(CONTEXTS);
const ctxMask = new Uint32Array(CONTEXTS);

function getSymInfo(): SymInfo[] {
    if (symInfo !== null) return symInfo;
    symInfo = [0, 1, 2].map((depth) => {
        const rawBits = new Uint8Array(TABLE_LEN);
        const dyCtxBase = new Uint8Array(TABLE_LEN);
        const valid = new Uint8Array(TABLE_LEN);
        const headShifted = new Int32Array(TABLE_LEN);
        const signMask = new Int32Array(TABLE_LEN);
        valid[0] = 1;
        dyCtxBase[0] = 2;
        for (let s = 1; s < SYMBOLS; s++) {
            const code = (s + 1) >>> 1;
            const len = code >>> depth;
            if (len < 1 || len > 32) continue;
            const kept = Math.min(len - 1, depth);
            const mantissa = code & lowMask(depth);
            if ((mantissa & lowMask(depth - kept)) !== 0) continue;
            const isNeg = (s & 1) === 0;
            if (len === 32 && (!isNeg || mantissa !== 0)) continue;
            const rb = len - 1 - kept;
            const hd = (1 << kept) | (mantissa >>> (depth - kept));
            const sign = isNeg ? -1 : 0;
            valid[s] = 1;
            rawBits[s] = rb;
            dyCtxBase[s] = 2 + 2 * Math.min(len, 5);
            signMask[s] = sign;
            headShifted[s] = rb === 0 ? ((hd ^ sign) - sign) | 0 : (hd << rb) | 0;
        }
        return { rawBits, dyCtxBase, valid, headShifted, signMask };
    });
    return symInfo;
}

export function decodeRansVertices(
    data: Uint8Array,
    runOffsets: Uint32Array | undefined,
    numWords: number,
): Int32Array {
    if ((numWords & 1) !== 0) throw new Error("rans odd");
    const vertexCount = numWords >>> 1;
    checkScratchBytes(vertexCount);
    let first: Uint8Array;
    if (vertexCount <= sharedFirst.length) {
        first = sharedFirst.subarray(0, vertexCount);
        first.fill(0);
    } else {
        first = new Uint8Array(vertexCount);
        if (vertexCount <= MAX_RETAINED_FIRST) sharedFirst = first;
    }
    if (runOffsets === undefined) {
        first.fill(1);
    } else {
        if (
            (runOffsets.length > 0 && runOffsets[0] !== 0) ||
            (runOffsets[runOffsets.length - 1] ?? 0) !== vertexCount
        ) {
            throw new Error("rans runs");
        }
        for (let i = 0; i < runOffsets.length; i++) {
            const o = runOffsets[i];
            if (o < vertexCount) first[o] = 1;
            else if (o !== vertexCount) throw new Error("rans runs");
        }
    }

    if (data.length === 0 || data[0] > 2) throw new Error("rans hdr");
    const { rawBits, dyCtxBase, valid, headShifted, signMask } = getSymInfo()[data[0]];
    const dLen = data.length;
    const tBitMax = (dLen - 1) * 8;
    let tBit = 0;

    const getBits = (bits: number): number => {
        if (bits === 0) return 0;
        const bi = 1 + (tBit >>> 3);
        const w =
            (bi < dLen ? data[bi] : 0) |
            (bi + 1 < dLen ? data[bi + 1] << 8 : 0) |
            (bi + 2 < dLen ? data[bi + 2] << 16 : 0);
        const res = (w >>> (tBit & 7)) & ((1 << bits) - 1);
        tBit += bits;
        return res;
    };
    const getGamma = (): number => {
        for (let z = 0; z <= 12; z++) {
            if (getBits(1) === 1) return (1 << z) | getBits(z);
        }
        throw new Error("rans freq");
    };

    ctxStart.fill(0);
    ctxPrecision.fill(0);
    ctxMask.fill(0);
    let slab = sharedSlab;
    let at = 1;

    for (let c = 0; c < CONTEXTS; c++) {
        const present = getBits(1);
        if (tBit > tBitMax) throw new Error("rans trunc");
        if (present === 0) continue;
        const precision = getBits(4);
        const max = getBits(9);
        if (precision > 11 || max >= SYMBOLS) throw new Error("rans table");
        let sum = 0;
        let badTbl = 0;
        if (precision === 0) {
            sharedFreq[max] = 1;
            for (let s = 0; s < max; s++) sharedFreq[s] = 0;
            sum = 1;
            if (!valid[max]) badTbl = 1;
        } else {
            for (let s = 0; s <= max; s++) {
                const f = getGamma() - 1;
                sharedFreq[s] = f;
                sum += f;
                if (f > 0 && !valid[s]) badTbl = 1;
            }
        }
        const slots = 1 << precision;
        if (tBit > tBitMax || sum !== slots || sharedFreq[max] === 0 || badTbl) {
            throw new Error("rans freq");
        }
        const needed = at + slots;
        if (needed > slab.length) {
            let nextSize = slab.length;
            while (nextSize < needed) nextSize <<= 1;
            checkScratchBytes(nextSize * 4);
            const grown = new Uint32Array(nextSize);
            grown.set(slab.subarray(0, at));
            slab = sharedSlab = grown;
        } else {
            checkScratchBytes(needed * 4);
        }
        ctxStart[c] = at;
        ctxPrecision[c] = precision;
        ctxMask[c] = slots - 1;
        for (let s = 0; s <= max; s++) {
            const f = sharedFreq[s];
            if (f === 0) continue;
            const base = ((f << 11) | (s << 23)) >>> 0;
            for (let k = 0; k < f; k++) slab[at + k] = (base + k) >>> 0;
            at += f;
        }
    }

    let slabSize = 4096;
    while (slabSize < at) slabSize <<= 1;
    checkScratchBytes(slabSize * 4);
    slab[0] = ((1 << 11) | (INVALID_SYM << 23)) >>> 0;

    const r = cursor(data, 1 + ((tBit + 7) >>> 3), dLen);
    const l0 = readVarintU32(r);
    const l1 = readVarintU32(r);
    const l2 = readVarintU32(r);
    const l3 = readVarintU32(r);
    const readLane = (len: number): [pos: number, end: number, state: number] => {
        if (len < 4 || (len & 1) !== 0 || r.e - r.p < len) throw new Error("rans lane");
        const start = r.p;
        readBytes(r, len);
        const st = (data[start] | (data[start + 1] << 8) | (data[start + 2] << 16) | (data[start + 3] << 24)) >>> 0;
        return [start + 4, start + len, st];
    };
    const lane0 = readLane(l0);
    const lane1 = readLane(l1);
    const lane2 = readLane(l2);
    const lane3 = readLane(l3);
    let p0 = lane0[0];
    let p1 = lane1[0];
    let p2 = lane2[0];
    let p3 = lane3[0];
    const e0 = lane0[1];
    const e1 = lane1[1];
    const e2 = lane2[1];
    const e3 = lane3[1];
    let s0 = lane0[2];
    let s1 = lane1[2];
    let s2 = lane2[2];
    let s3 = lane3[2];

    const rawStart = r.p;
    const rawEnd = r.e;
    const rawFastEnd = rawEnd - 5;
    let rawBit = 0;
    const slabMask = slabSize - 1;

    const readRawTail = (bi: number, sh: number, rb: number): number => {
        const lo32 =
            ((bi < rawEnd ? data[bi] : 0) |
                (bi + 1 < rawEnd ? data[bi + 1] << 8 : 0) |
                (bi + 2 < rawEnd ? data[bi + 2] << 16 : 0) |
                (bi + 3 < rawEnd ? data[bi + 3] << 24 : 0)) >>>
            0;
        const b4 = bi + 4 < rawEnd ? data[bi + 4] : 0;
        const mask = 0xffffffff >>> (32 - rb);
        return sh === 0 ? (lo32 & mask) >>> 0 : (((lo32 >>> sh) | (b4 << (32 - sh))) & mask) >>> 0;
    };

    const out = numWords === 0 ? EMPTY_I32 : allocI32(numWords);
    let x = 0;
    let y = 0;
    let bad = 0;

    for (let v = 0; v < vertexCount; v += 2) {
        const f0 = first[v];
        const eX0 = slab[(ctxStart[f0] + (s0 & ctxMask[f0])) & slabMask];
        s0 = (Math.imul((eX0 >>> 11) & 0xfff, s0 >>> ctxPrecision[f0]) + (eX0 & 0x7ff)) >>> 0;
        if (s0 < STATE_LOW) {
            s0 = ((s0 << 16) | (p0 + 2 <= e0 ? data[p0] | (data[p0 + 1] << 8) : 0)) >>> 0;
            p0 += 2;
        }
        const symX0 = eX0 >>> 23;

        const hasPair = v + 1 < vertexCount;
        let symX1 = 0;
        let f1 = 0;
        if (hasPair) {
            f1 = first[v + 1];
            const eX1 = slab[(ctxStart[f1] + (s2 & ctxMask[f1])) & slabMask];
            s2 = (Math.imul((eX1 >>> 11) & 0xfff, s2 >>> ctxPrecision[f1]) + (eX1 & 0x7ff)) >>> 0;
            if (s2 < STATE_LOW) {
                s2 = ((s2 << 16) | (p2 + 2 <= e2 ? data[p2] | (data[p2 + 1] << 8) : 0)) >>> 0;
                p2 += 2;
            }
            symX1 = eX1 >>> 23;
        }

        const cY0 = dyCtxBase[symX0] + f0;
        const eY0 = slab[(ctxStart[cY0] + (s1 & ctxMask[cY0])) & slabMask];
        s1 = (Math.imul((eY0 >>> 11) & 0xfff, s1 >>> ctxPrecision[cY0]) + (eY0 & 0x7ff)) >>> 0;
        if (s1 < STATE_LOW) {
            s1 = ((s1 << 16) | (p1 + 2 <= e1 ? data[p1] | (data[p1 + 1] << 8) : 0)) >>> 0;
            p1 += 2;
        }
        const symY0 = eY0 >>> 23;

        let symY1 = 0;
        if (hasPair) {
            const cY1 = dyCtxBase[symX1] + f1;
            const eY1 = slab[(ctxStart[cY1] + (s3 & ctxMask[cY1])) & slabMask];
            s3 = (Math.imul((eY1 >>> 11) & 0xfff, s3 >>> ctxPrecision[cY1]) + (eY1 & 0x7ff)) >>> 0;
            if (s3 < STATE_LOW) {
                s3 = ((s3 << 16) | (p3 + 2 <= e3 ? data[p3] | (data[p3 + 1] << 8) : 0)) >>> 0;
                p3 += 2;
            }
            symY1 = eY1 >>> 23;
        }

        bad |= (256 - symX0) | (256 - symY0);
        const rbX0 = rawBits[symX0];
        if (rbX0 === 0) {
            x = (x + headShifted[symX0]) | 0;
        } else {
            const bi = rawStart + (rawBit >>> 3);
            const sh = rawBit & 7;
            const mask = 0xffffffff >>> (32 - rbX0);
            const low =
                bi <= rawFastEnd
                    ? sh === 0
                        ? ((data[bi] | (data[bi + 1] << 8) | (data[bi + 2] << 16) | (data[bi + 3] << 24)) & mask) >>> 0
                        : (((((data[bi] | (data[bi + 1] << 8) | (data[bi + 2] << 16) | (data[bi + 3] << 24)) >>> 0) >>>
                              sh) |
                              (data[bi + 4] << (32 - sh))) &
                              mask) >>>
                          0
                    : readRawTail(bi, sh, rbX0);
            rawBit += rbX0;
            const sign = signMask[symX0];
            x = (x + (((headShifted[symX0] | low) ^ sign) - sign)) | 0;
        }

        const rbY0 = rawBits[symY0];
        if (rbY0 === 0) {
            y = (y + headShifted[symY0]) | 0;
        } else {
            const bi = rawStart + (rawBit >>> 3);
            const sh = rawBit & 7;
            const mask = 0xffffffff >>> (32 - rbY0);
            const low =
                bi <= rawFastEnd
                    ? sh === 0
                        ? ((data[bi] | (data[bi + 1] << 8) | (data[bi + 2] << 16) | (data[bi + 3] << 24)) & mask) >>> 0
                        : (((((data[bi] | (data[bi + 1] << 8) | (data[bi + 2] << 16) | (data[bi + 3] << 24)) >>> 0) >>>
                              sh) |
                              (data[bi + 4] << (32 - sh))) &
                              mask) >>>
                          0
                    : readRawTail(bi, sh, rbY0);
            rawBit += rbY0;
            const sign = signMask[symY0];
            y = (y + (((headShifted[symY0] | low) ^ sign) - sign)) | 0;
        }
        out[v * 2] = x;
        out[v * 2 + 1] = y;

        if (hasPair) {
            bad |= (256 - symX1) | (256 - symY1);
            const rbX1 = rawBits[symX1];
            if (rbX1 === 0) {
                x = (x + headShifted[symX1]) | 0;
            } else {
                const bi = rawStart + (rawBit >>> 3);
                const sh = rawBit & 7;
                const mask = 0xffffffff >>> (32 - rbX1);
                const low =
                    bi <= rawFastEnd
                        ? sh === 0
                            ? ((data[bi] | (data[bi + 1] << 8) | (data[bi + 2] << 16) | (data[bi + 3] << 24)) &
                                  mask) >>>
                              0
                            : (((((data[bi] | (data[bi + 1] << 8) | (data[bi + 2] << 16) | (data[bi + 3] << 24)) >>>
                                  0) >>>
                                  sh) |
                                  (data[bi + 4] << (32 - sh))) &
                                  mask) >>>
                              0
                        : readRawTail(bi, sh, rbX1);
                rawBit += rbX1;
                const sign = signMask[symX1];
                x = (x + (((headShifted[symX1] | low) ^ sign) - sign)) | 0;
            }

            const rbY1 = rawBits[symY1];
            if (rbY1 === 0) {
                y = (y + headShifted[symY1]) | 0;
            } else {
                const bi = rawStart + (rawBit >>> 3);
                const sh = rawBit & 7;
                const mask = 0xffffffff >>> (32 - rbY1);
                const low =
                    bi <= rawFastEnd
                        ? sh === 0
                            ? ((data[bi] | (data[bi + 1] << 8) | (data[bi + 2] << 16) | (data[bi + 3] << 24)) &
                                  mask) >>>
                              0
                            : (((((data[bi] | (data[bi + 1] << 8) | (data[bi + 2] << 16) | (data[bi + 3] << 24)) >>>
                                  0) >>>
                                  sh) |
                                  (data[bi + 4] << (32 - sh))) &
                                  mask) >>>
                              0
                        : readRawTail(bi, sh, rbY1);
                rawBit += rbY1;
                const sign = signMask[symY1];
                y = (y + (((headShifted[symY1] | low) ^ sign) - sign)) | 0;
            }
            out[v * 2 + 2] = x;
            out[v * 2 + 3] = y;
        }
    }

    if (bad < 0) throw new Error("rans missing table");
    if (
        s0 !== STATE_LOW ||
        s1 !== STATE_LOW ||
        s2 !== STATE_LOW ||
        s3 !== STATE_LOW ||
        p0 !== e0 ||
        p1 !== e1 ||
        p2 !== e2 ||
        p3 !== e3
    ) {
        throw new Error("rans state");
    }
    if ((rawBit + 7) >>> 3 !== rawEnd - rawStart) throw new Error("rans raw len");
    return out;
}
