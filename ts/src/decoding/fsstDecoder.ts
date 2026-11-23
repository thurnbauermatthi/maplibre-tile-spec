const ESCAPE_MARKER = 255;
const MAX_SYMBOL_LEN = 8;

/**
 * Decode FSST compressed data.
 *
 * @param symbols           Concatenated symbol bytes (each symbol 0..8 bytes)
 * @param symbolLengths     Byte length of each symbol
 * @param compressedData    FSST-compressed corpus bytes
 * @param onAlloc           Optional budget callback invoked with decoded byte length before allocation
 */
export function decodeFsst(
    symbols: Uint8Array,
    symbolLengths: Uint32Array,
    compressedData: Uint8Array,
    onAlloc?: (bytes: number) => void,
): Uint8Array {
    const symCount = symbolLengths.length;
    const usableCount = Math.min(symCount, ESCAPE_MARKER);
    const symbolOffsets = new Uint32Array(usableCount);
    let symTotal = 0;
    for (let i = 0; i < symCount; i++) {
        const len = symbolLengths[i];
        if (len > MAX_SYMBOL_LEN) {
            throw new RangeError(`FSST symbol ${i} length ${len} exceeds ${MAX_SYMBOL_LEN}`);
        }
        if (i < usableCount) symbolOffsets[i] = symTotal;
        symTotal += len;
        if (symTotal > symbols.length) {
            throw new RangeError(`FSST symbol lengths (${symTotal}) overrun symbol table (${symbols.length})`);
        }
    }

    const cLen = compressedData.length;
    let decodedLength = 0;
    for (let i = 0; i < cLen; i++) {
        const code = compressedData[i];
        if (code === ESCAPE_MARKER) {
            if (++i >= cLen) throw new RangeError("FSST corpus ends on an escape marker");
            decodedLength++;
        } else {
            if (code >= usableCount) {
                throw new RangeError(`FSST symbol index ${code} out of bounds (${usableCount})`);
            }
            decodedLength += symbolLengths[code];
        }
    }

    onAlloc?.(decodedLength);
    const decodedData = new Uint8Array(decodedLength);
    let outPos = 0;
    for (let i = 0; i < cLen; i++) {
        const code = compressedData[i];
        if (code === ESCAPE_MARKER) {
            decodedData[outPos++] = compressedData[++i];
        } else {
            let len = symbolLengths[code];
            let src = symbolOffsets[code];
            while (len-- > 0) {
                decodedData[outPos++] = symbols[src++];
            }
        }
    }

    return decodedData;
}
