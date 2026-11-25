import { type ByteReader, allocU8, readBytes, readU8 } from "./binary";

export const PRES_BITMAP = 0;
export const PRES_BYTE_RLE = 1;

const POP = new Uint8Array(256);
for (let i = 1; i < 256; i++) POP[i] = POP[i >> 1] + (i & 1);

export function maskBitmapTail(bytes: Uint8Array, count: number): Uint8Array {
    const byteLen = (count + 7) >>> 3;
    const rem = count & 7;
    if (rem === 0) return bytes.length === byteLen ? bytes : bytes.subarray(0, byteLen);
    const mask = (1 << rem) - 1;
    if ((bytes[byteLen - 1] & ~mask) === 0 && bytes.length === byteLen) return bytes;
    const copy = bytes.slice(0, byteLen);
    copy[byteLen - 1] &= mask;
    return copy;
}

export function bitmapPopcount(bitmap: Uint8Array, count: number): number {
    const full = count >>> 3;
    let ones = 0;
    for (let i = 0; i < full; i++) ones += POP[bitmap[i]];
    const rem = count & 7;
    if (rem > 0) ones += POP[bitmap[full] & ((1 << rem) - 1)];
    return ones;
}

export function isBitSet(bitmap: Uint8Array | undefined, i: number): boolean {
    return bitmap === undefined || ((bitmap[i >>> 3] >>> (i & 7)) & 1) === 1;
}

export function normalizePresentBitmap(bitmap: Uint8Array | undefined, count: number): Uint8Array | undefined {
    if (bitmap === undefined || count === 0) return undefined;
    return bitmapPopcount(bitmap, count) === count ? undefined : maskBitmapTail(bitmap, count);
}

export interface MutableIndexable<T> {
    [index: number]: T;
}

export function scatterPresent<T>(
    bitmap: Uint8Array,
    count: number,
    dense: ArrayLike<T>,
    target: MutableIndexable<T>,
): void {
    let d = 0;
    const full = count >>> 3;
    for (let bi = 0; bi < full; bi++) {
        let b = bitmap[bi];
        if (b === 0) continue;
        const base = bi << 3;
        if (b === 0xff) {
            target[base] = dense[d++];
            target[base + 1] = dense[d++];
            target[base + 2] = dense[d++];
            target[base + 3] = dense[d++];
            target[base + 4] = dense[d++];
            target[base + 5] = dense[d++];
            target[base + 6] = dense[d++];
            target[base + 7] = dense[d++];
            continue;
        }
        while (b !== 0) {
            const lsb = b & -b;
            target[base + 31 - Math.clz32(lsb)] = dense[d++];
            b ^= lsb;
        }
    }
    const rem = count & 7;
    if (rem > 0) {
        let b = bitmap[full] & ((1 << rem) - 1);
        const base = full << 3;
        while (b !== 0) {
            const lsb = b & -b;
            target[base + 31 - Math.clz32(lsb)] = dense[d++];
            b ^= lsb;
        }
    }
}

export function readPresenceBits(r: ByteReader, count: number, coding: number): Uint8Array {
    const byteLen = (count + 7) >>> 3;
    if (coding === PRES_BITMAP) return maskBitmapTail(readBytes(r, byteLen), count);
    if (coding === PRES_BYTE_RLE) {
        const plain = allocU8(byteLen);
        let produced = 0;
        while (produced < byteLen) {
            const control = readU8(r);
            if (control >= 128) {
                const len = 256 - control;
                if (produced + len > byteLen) throw new Error("presence rle overflow");
                plain.set(readBytes(r, len), produced);
                produced += len;
            } else {
                const len = control + 3;
                const val = readU8(r);
                if (produced + len > byteLen) throw new Error("presence rle overflow");
                plain.fill(val, produced, produced + len);
                produced += len;
            }
        }
        return maskBitmapTail(plain, count);
    }
    throw new Error("presence coding");
}
