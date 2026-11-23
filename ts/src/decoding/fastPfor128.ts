import { createDecoderWorkspace, decodeFastPforInt32 } from "./fastPforDecoder";

const IS_LE = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;
const sharedWorkspace = createDecoderWorkspace();
const MAX_RETAINED_WORDS = 65536;
let sharedWords = new Uint32Array(1024);
let sharedBytes = new Uint8Array(sharedWords.buffer);

/**
 * The v2 FastPFOR variant (`docs/encodings.md#fastpfor`): 128-value blocks over little-endian
 * words. The `fastpfor` crate mlt-core decodes with packs every block as four sequential 32-value
 * groups, exactly as the v1 kernel does for eight, so the v1 decoder runs with the block size
 * changed. 64-bit words have no v2 FastPFOR form in the reference decoder.
 */
export function decodeFastPfor128(
    bytes: Uint8Array,
    start: number,
    byteLength: number,
    numValues: number,
    out?: Uint32Array,
): Uint32Array {
    if ((byteLength & 3) !== 0) {
        throw new Error(`FastPFOR payload of ${byteLength} bytes is not whole 32-bit words`);
    }
    if (numValues === 0) {
        if (byteLength !== 0) {
            throw new Error(`FastPFOR stream of no values carries ${byteLength} bytes`);
        }
        return out ?? new Uint32Array(0);
    }
    const wordCount = byteLength >>> 2;
    const byteStart = bytes.byteOffset + start;
    let words: Uint32Array;
    if (IS_LE && (byteStart & 3) === 0) {
        words = new Uint32Array(bytes.buffer, byteStart, wordCount);
    } else if (IS_LE) {
        let buf = sharedWords;
        let u8 = sharedBytes;
        if (wordCount > buf.length) {
            buf = new Uint32Array(wordCount);
            u8 = new Uint8Array(buf.buffer);
            if (wordCount <= MAX_RETAINED_WORDS) {
                sharedWords = buf;
                sharedBytes = u8;
            }
        }
        u8.set(bytes.subarray(start, start + byteLength));
        words = buf.subarray(0, wordCount);
    } else {
        const view = new DataView(bytes.buffer, byteStart, byteLength);
        words = new Uint32Array(wordCount);
        for (let i = 0; i < wordCount; i++) {
            words[i] = view.getUint32(i * 4, true);
        }
    }
    return decodeFastPforInt32(words, numValues, sharedWorkspace, 128, out);
}
