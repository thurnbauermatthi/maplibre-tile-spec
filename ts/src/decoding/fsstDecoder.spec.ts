import { describe, it, expect } from "vitest";
import { decodeFsst } from "./fsstDecoder";
import { encodeFsst, createSymbolTable } from "../encoding/fsstEncoder";

const textEncoder = new TextEncoder();

describe("decodeFsst", () => {
    describe("basic functionality", () => {
        it("should decode FSST compressed string data", () => {
            const inputString = "HelloWorld !";
            const originalBytes = textEncoder.encode(inputString);

            const { symbols, symbolLengths } = createSymbolTable(["Hello", "World", "!"]);
            const encoded = encodeFsst(symbols, symbolLengths, originalBytes);
            const decoded = decodeFsst(symbols, symbolLengths, encoded);

            expect(decoded).toEqual(originalBytes);
            expect(new TextDecoder().decode(decoded)).toBe(inputString);
        });

        it("should handle empty string", () => {
            const inputString = "";
            const originalBytes = textEncoder.encode(inputString);

            const { symbols, symbolLengths } = createSymbolTable(["A"]);
            const encoded = encodeFsst(symbols, symbolLengths, originalBytes);
            const decoded = decodeFsst(symbols, symbolLengths, encoded);

            expect(decoded).toEqual(originalBytes);
            expect(decoded.length).toBe(0);
        });

        it("should handle string with no matching symbols", () => {
            const inputString = "12345";
            const originalBytes = textEncoder.encode(inputString);

            const { symbols, symbolLengths } = createSymbolTable(["abc", "def", "xyz"]);
            const encoded = encodeFsst(symbols, symbolLengths, originalBytes);
            const decoded = decodeFsst(symbols, symbolLengths, encoded);

            expect(decoded).toEqual(originalBytes);
            expect(new TextDecoder().decode(decoded)).toBe(inputString);
        });

        it("should handle consecutive escaped bytes", () => {
            const symbols = new Uint8Array([65]);
            const symbolLengths = new Uint32Array([1]);
            const encoded = new Uint8Array([255, 1, 255, 2, 0, 255, 3]);

            expect(decodeFsst(symbols, symbolLengths, encoded)).toEqual(new Uint8Array([1, 2, 65, 3]));
        });

        it("should handle string with all matching symbols", () => {
            const inputString = "AAAA";
            const originalBytes = textEncoder.encode(inputString);

            const { symbols, symbolLengths } = createSymbolTable(["A"]);
            const encoded = encodeFsst(symbols, symbolLengths, originalBytes);
            const decoded = decodeFsst(symbols, symbolLengths, encoded);

            expect(decoded).toEqual(originalBytes);
            expect(new TextDecoder().decode(decoded)).toBe(inputString);
        });
    });

    describe("compression verification", () => {
        it("should handle repeated strings efficiently", () => {
            const inputString = "HelloWorldHelloWorld";
            const originalBytes = textEncoder.encode(inputString);

            const { symbols, symbolLengths } = createSymbolTable(["Hello", "World"]);
            const encoded = encodeFsst(symbols, symbolLengths, originalBytes);
            const decoded = decodeFsst(symbols, symbolLengths, encoded);

            expect(decoded).toEqual(originalBytes);
            expect(new TextDecoder().decode(decoded)).toBe(inputString);
            expect(encoded.length).toBeLessThan(originalBytes.length);
        });
    });

    describe("malformed input validation", () => {
        it("rejects symbol length > 8, symbol table overrun, trailing escape, and out-of-range symbol codes", () => {
            expect(() => decodeFsst(new Uint8Array(9), new Uint32Array([9]), new Uint8Array([0]))).toThrow(RangeError);
            expect(() => decodeFsst(new Uint8Array(3), new Uint32Array([4]), new Uint8Array([0]))).toThrow(RangeError);
            expect(() => decodeFsst(new Uint8Array([65]), new Uint32Array([1]), new Uint8Array([255]))).toThrow(
                RangeError,
            );
            expect(() => decodeFsst(new Uint8Array([65]), new Uint32Array([1]), new Uint8Array([0, 255]))).toThrow(
                RangeError,
            );
            expect(() => decodeFsst(new Uint8Array([65]), new Uint32Array([1]), new Uint8Array([1]))).toThrow(
                RangeError,
            );
        });

        it("invokes onAlloc with exact decoded byte length before allocating", () => {
            const symbols = new Uint8Array([65, 66, 67]);
            const symbolLengths = new Uint32Array([2, 1]);
            let charged = -1;
            const out = decodeFsst(symbols, symbolLengths, new Uint8Array([0, 255, 68, 1]), (bytes) => {
                charged = bytes;
            });
            expect(charged).toBe(4);
            expect(out).toEqual(new Uint8Array([65, 66, 68, 67]));
        });
    });
});
