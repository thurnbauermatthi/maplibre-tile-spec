import { type ByteReader, allocI32, chargeBytes, checkScratchBytes, cursor, readBytes, readVarintU64 } from "./binary";
import { GEOM_LINESTRING, GEOM_MULTILINESTRING, GEOM_MULTIPOINT, GEOM_POINT, type TopologyLevels } from "./geometry";

export const RUN_POINT = 1;
export const RUN_LINE = 2;
export const RUN_RING = 3;

const FLAG_COPIES = 1;
const FLAG_FROM_END = 4;
const MIN_COPY = 2;

const CL_START = 0;
const CL_POINT = 1;
const CL_RING = 2;
const CL_CORNER = 3;
const CL_LINE1 = 4;
const CL_LINE = 5;
const CLASSES = 6;

export interface PredictedRuns {
    lengths: Uint32Array;
    kinds: Uint8Array;
}

export function predictedRuns(types: Uint8Array, levels: TopologyLevels, vertexCount: number): PredictedRuns {
    const finest = levels.ringOffsets ?? levels.partOffsets ?? levels.geometryOffsets;
    if (finest === undefined) {
        if (types.length !== vertexCount) throw new Error("predicted: vertices without topology");
        for (let i = 0; i < types.length; i++) {
            const t = types[i];
            if (t !== GEOM_POINT && t !== GEOM_MULTIPOINT) throw new Error("predicted: vertices without topology");
        }
        checkScratchBytes(vertexCount * 5);
        return {
            lengths: new Uint32Array(vertexCount).fill(1),
            kinds: new Uint8Array(vertexCount).fill(RUN_POINT),
        };
    }

    const coarser: Uint32Array[] = [];
    if (levels.geometryOffsets !== undefined && levels.geometryOffsets !== finest) coarser.push(levels.geometryOffsets);
    if (levels.partOffsets !== undefined && levels.partOffsets !== finest) coarser.push(levels.partOffsets);

    const runCount = finest.length > 0 ? finest.length - 1 : 0;
    checkScratchBytes(runCount * 5);
    const kinds = new Uint8Array(runCount);

    for (let f = 0; f < types.length; f++) {
        let rStart = f;
        let rEnd = f + 1;
        for (let l = 0; l < coarser.length; l++) {
            const level = coarser[l];
            if (rStart >= level.length || rEnd >= level.length) throw new Error("predicted: topology out of range");
            const s = level[rStart];
            const e = level[rEnd];
            if (s > e) throw new Error("predicted: topology out of range");
            rStart = s;
            rEnd = e;
        }
        if (rEnd > runCount) throw new Error("predicted: topology out of range");
        const t = types[f];
        const kind =
            t === GEOM_POINT || t === GEOM_MULTIPOINT
                ? RUN_POINT
                : t === GEOM_LINESTRING || t === GEOM_MULTILINESTRING
                  ? RUN_LINE
                  : RUN_RING;
        kinds.fill(kind, rStart, rEnd);
    }

    const lengths = new Uint32Array(runCount);
    for (let r = 0; r < runCount; r++) {
        if (kinds[r] === 0) throw new Error("predicted: unowned run");
        const a = finest[r];
        const b = finest[r + 1];
        if (b < a) throw new Error("predicted: decreasing run offsets");
        lengths[r] = b - a;
    }
    const covered = finest.length > 0 ? finest[finest.length - 1] : 0;
    if (covered !== vertexCount) throw new Error("predicted: run vertex count mismatch");
    return { lengths, kinds };
}

function takeSubSlice(r: ByteReader, lenBig: bigint): Uint8Array {
    if (lenBig > BigInt(r.e - r.p)) throw new RangeError("predicted: block past payload");
    return readBytes(r, Number(lenBig));
}

function unpackPlane(lo: Uint8Array, hi: Uint8Array): Float64Array {
    const loLen = lo.length;
    chargeBytes(loLen * 8);
    const out = new Float64Array(loLen);
    let hiPos = 0;
    for (let i = 0; i < loLen; i++) {
        const b0 = lo[i];
        if (b0 < 255) {
            out[i] = (b0 >>> 1) ^ -(b0 & 1);
            continue;
        }
        if (hiPos >= hi.length) throw new RangeError("predicted: escaped eof");
        const c0 = hi[hiPos++];
        if (c0 < 128) {
            const z = c0 + 255;
            out[i] = (z >>> 1) ^ -(z & 1);
            continue;
        }
        let v = BigInt(c0 & 127);
        let shift = 7n;
        while (true) {
            if (shift > 56n) throw new RangeError("predicted: escaped overflow");
            if (hiPos >= hi.length) throw new RangeError("predicted: escaped eof");
            const c = hi[hiPos++];
            v |= BigInt(c & 127) << shift;
            if (c < 128) {
                if (c === 0) throw new Error("predicted: non-canonical varint");
                break;
            }
            shift += 7n;
        }
        const z = v + 255n;
        const r = (z >> 1n) ^ -(z & 1n);
        out[i] = r > 1_000_000_000_000n ? 1e12 : r < -1_000_000_000_000n ? -1e12 : Number(r);
    }
    if (hiPos !== hi.length) throw new Error("predicted: trailing escaped");
    return out;
}

function stepSource(
    k: number,
    backwards: boolean,
    written: number,
    srcStart: number,
    srcLen: number,
    srcClosed: boolean,
): number {
    const complete = srcStart + srcLen <= written;
    const last = srcStart + srcLen - 1;
    let next: number;
    if (backwards) {
        if (k > srcStart) next = k - 1;
        else if (srcClosed && complete) next = last;
        else return -1;
    } else {
        if (k < last) next = k + 1;
        else if (srcClosed && complete) next = srcStart;
        else return -1;
    }
    return next < written ? next : -1;
}

function findRunIndex(starts: Uint32Array, k: number): number {
    let lo = 0;
    let hi = starts.length;
    while (lo < hi) {
        const mid = (lo + hi) >>> 1;
        if (starts[mid] <= k) lo = mid + 1;
        else hi = mid;
    }
    return lo - 1;
}

export function decodePredictedRuns(data: Uint8Array, runs: PredictedRuns, vertexCount: number): Int32Array {
    const words = vertexCount * 2;
    if (!Number.isSafeInteger(words) || words < 0) throw new RangeError("predicted: vertex count overflow");
    const v = allocI32(words);

    if (data.length === 0) throw new RangeError("predicted: empty payload");
    const flags = data[0];
    if ((flags & ~(FLAG_COPIES | FLAG_FROM_END)) !== 0) throw new Error("predicted: reserved flags");

    const fromEnd = (flags & FLAG_FROM_END) !== 0;
    const flagsCopies = (flags & FLAG_COPIES) !== 0;
    const inCur = cursor(data, 1, data.length);
    const ops = flagsCopies ? cursor(takeSubSlice(inCur, readVarintU64(inCur))) : cursor(data, 0, 0);

    const residuals = new Array<Float64Array>(CLASSES);
    for (let c = 0; c < CLASSES; c++) {
        const blockBytes = takeSubSlice(inCur, readVarintU64(inCur));
        const blockCur = cursor(blockBytes);
        const lo = takeSubSlice(blockCur, readVarintU64(blockCur));
        residuals[c] = unpackPlane(lo, blockBytes.subarray(blockCur.p, blockCur.e));
    }
    if (inCur.p !== inCur.e) throw new Error("predicted: trailing bytes");

    const { lengths, kinds } = runs;
    const runCount = lengths.length;
    let total = 0;
    for (let r = 0; r < runCount; r++) total += lengths[r];
    if (total !== vertexCount) throw new Error("predicted: run vertex count mismatch");

    let starts: Uint32Array | null = null;
    if (flagsCopies) {
        checkScratchBytes((runCount + 1) * 4);
        starts = new Uint32Array(runCount + 1);
        let acc = 0;
        for (let r = 0; r < runCount; r++) {
            starts[r] = acc;
            acc += lengths[r];
        }
        starts[runCount] = acc;
    }

    const cursors = new Int32Array(CLASSES);
    let prevX = 0;
    let prevY = 0;
    let prevStartX = 0;
    let prevStartY = 0;
    let s = 0;

    for (let r = 0; r < runCount; r++) {
        const len = lengths[r];
        const kind = kinds[r];

        let copies = 0;
        if (flagsCopies && kind !== RUN_POINT) {
            const cBig = readVarintU64(ops);
            copies = cBig > BigInt(len + 1) ? len + 1 : Number(cBig);
        }
        let nextCopy = -1;
        if (copies > 0) {
            const gapBig = readVarintU64(ops);
            nextCopy = gapBig > BigInt(len + 1) ? len + 1 : Number(gapBig);
        }

        let i = 0;
        while (i < len) {
            if (nextCopy === 0) {
                const packed = readVarintU64(ops);
                const distBig = packed >> 1n;
                const backwards = (packed & 1n) === 1n;
                const nBig = readVarintU64(ops);
                const here = s + i;
                if (distBig === 0n || distBig > BigInt(here)) {
                    throw new RangeError("predicted: copy before first vertex");
                }
                if (nBig > BigInt(len - i - MIN_COPY)) {
                    throw new RangeError("predicted: copy past run end");
                }
                const n = Number(nBig) + MIN_COPY;
                let k = here - Number(distBig);
                const srcIdx = findRunIndex(starts, k);
                const srcStart = starts[srcIdx];
                const srcLen = starts[srcIdx + 1] - srcStart;
                const srcClosed = kinds[srcIdx] === RUN_RING;

                for (let c = 0; c < n; c++) {
                    const written = here + c;
                    if (k >= written) throw new RangeError("predicted: copy unwritten vertex");
                    v[2 * written] = v[2 * k];
                    v[2 * written + 1] = v[2 * k + 1];
                    if (c + 1 < n) {
                        k = stepSource(k, backwards, written + 1, srcStart, srcLen, srcClosed);
                        if (k < 0) throw new RangeError("predicted: copy past source run");
                    }
                }
                i += n;
                copies--;
                if (copies > 0) {
                    const gapBig = readVarintU64(ops);
                    nextCopy = gapBig > BigInt(len + 1) ? len + 1 : Number(gapBig);
                } else {
                    nextCopy = -1;
                }
                continue;
            }

            const w = s + i;
            let px: number;
            let py: number;
            if (i === 0) {
                const cls = kind === RUN_POINT ? CL_POINT : CL_START;
                const plane = residuals[cls];
                const cur = cursors[cls];
                if (cur + 2 > plane.length) throw new RangeError("predicted: residual underflow");
                cursors[cls] = cur + 2;
                const rx = plane[cur];
                const ry = plane[cur + 1];
                if (kind === RUN_POINT) {
                    px = prevX + rx;
                    py = prevY + ry;
                } else if (fromEnd) {
                    px = prevX + rx;
                    py = prevY + ry;
                } else {
                    px = prevStartX + rx;
                    py = prevStartY + ry;
                }
            } else {
                const cls =
                    kind === RUN_POINT
                        ? CL_POINT
                        : kind === RUN_RING
                          ? len === 4 && i === 3
                              ? CL_CORNER
                              : CL_RING
                          : i === 1
                            ? CL_LINE1
                            : CL_LINE;
                const plane = residuals[cls];
                const cur = cursors[cls];
                if (cur + 2 > plane.length) throw new RangeError("predicted: residual underflow");
                cursors[cls] = cur + 2;
                const rx = plane[cur];
                const ry = plane[cur + 1];
                if (cls === CL_CORNER) {
                    px = v[2 * s] + v[2 * (s + 2)] - v[2 * (s + 1)] + rx;
                    py = v[2 * s + 1] + v[2 * (s + 2) + 1] - v[2 * (s + 1) + 1] + ry;
                } else {
                    px = v[2 * (w - 1)] + rx;
                    py = v[2 * (w - 1) + 1] + ry;
                }
            }

            if (px < -2147483648 || px > 2147483647 || py < -2147483648 || py > 2147483647) {
                throw new RangeError("predicted: vertex out of range");
            }
            v[2 * w] = px;
            v[2 * w + 1] = py;
            i++;
            if (nextCopy >= 0) nextCopy--;
        }

        if (copies > 0 || nextCopy >= 0) throw new Error("predicted: pending copy past run end");
        if (len > 0) {
            if (kind !== RUN_POINT) {
                prevStartX = v[2 * s];
                prevStartY = v[2 * s + 1];
            }
            const lastIdx = 2 * (s + len - 1);
            prevX = v[lastIdx];
            prevY = v[lastIdx + 1];
        }
        s += len;
    }

    if (ops.p !== ops.e) throw new Error("predicted: trailing copy ops");
    for (let c = 0; c < CLASSES; c++) {
        if (cursors[c] !== residuals[c].length) throw new Error("predicted: trailing residuals");
    }
    return v;
}

export function decodePredictedVertices(
    data: Uint8Array,
    types: Uint8Array,
    levels: TopologyLevels,
    vertexCount: number,
): Int32Array {
    return decodePredictedRuns(data, predictedRuns(types, levels, vertexCount), vertexCount);
}
