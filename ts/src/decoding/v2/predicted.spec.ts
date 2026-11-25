import { readdirSync, readFileSync } from "node:fs";
import * as path from "node:path";
import { describe, expect, it } from "vitest";
import { compareWithTolerance } from "../../../../test/synthetic/synthetic-test-utils";
import { decodeTile, decodeTileColumns, tileGeoJson } from "../../index";
import { featureTablesToFeatureCollection } from "../../vector/featureTablesToGeoJson";
import { resetBudget } from "./binary";
import { GEOM_LINESTRING, GEOM_POINT, GEOM_POLYGON } from "./geometry";
import { RUN_LINE, RUN_RING, decodePredictedRuns, decodePredictedVertices, predictedRuns } from "./predicted";

const REPO_ROOT = path.resolve(__dirname, "../../../..");
const EXPECTED_02 = path.join(REPO_ROOT, "test/expected/0x02");

describe("Predicted vertex decoder", () => {
    expect.addEqualityTesters([compareWithTolerance]);

    it("decodes the byte-for-byte example of the v2 specification", () => {
        resetBudget();
        const payload = new Uint8Array([
            0x00, // flags
            0x05,
            0x04,
            0x14,
            0x14,
            0x13,
            0x13, // Start
            0x01,
            0x00, // Point
            0x05,
            0x04,
            0x14,
            0x00,
            0x00,
            0x14, // Ring
            0x03,
            0x02,
            0x00,
            0x00, // Corner
            0x03,
            0x02,
            0x0a,
            0x00, // Line1
            0x03,
            0x02,
            0x0a,
            0x02, // Line
        ]);
        const runs = {
            lengths: new Uint32Array([4, 3]),
            kinds: new Uint8Array([RUN_RING, RUN_LINE]),
        };
        const decoded = decodePredictedRuns(payload, runs, 7);
        expect(Array.from(decoded)).toEqual([10, 10, 20, 10, 20, 20, 10, 20, 0, 0, 5, 0, 10, 1]);
    });

    it("rejects reserved flag bits, trailing bytes, and mismatched run counts", () => {
        resetBudget();
        const valid = new Uint8Array([
            0x00,
            0x03,
            0x02,
            0x02,
            0x04, // Start: (1, 2)
            0x01,
            0x00, // Point
            0x01,
            0x00, // Ring
            0x01,
            0x00, // Corner
            0x03,
            0x02,
            0x04,
            0x04, // Line1: (+2, +2) -> (3, 4)
            0x01,
            0x00, // Line
        ]);
        const runs = { lengths: new Uint32Array([2]), kinds: new Uint8Array([RUN_LINE]) };
        expect(Array.from(decodePredictedRuns(valid, runs, 2))).toEqual([1, 2, 3, 4]);

        for (const bit of [1, 3, 4, 5, 6, 7]) {
            const bad = valid.slice();
            bad[0] = 1 << bit;
            expect(() => decodePredictedRuns(bad, runs, 2)).toThrow();
        }

        const trailing = new Uint8Array(valid.length + 1);
        trailing.set(valid);
        expect(() => decodePredictedRuns(trailing, runs, 2)).toThrow();
        expect(() => decodePredictedRuns(valid, runs, 3)).toThrow();
    });

    it("decodes backward copies across shared walls, closed-ring wraparound, and escaped i32 extremes", () => {
        resetBudget();
        // Ring 1: (0,0), (10,0), (10,10), (0,10)
        // Ring 2 copies 2 vertices backwards from index 2 ((10,10), (10,0)), then literals (20,0), (20,10)
        const payload = new Uint8Array([
            0x01, // FLAG_COPIES
            0x05, // ops len = 5
            0x00, // ring 1: 0 copies
            0x01,
            0x00,
            0x05,
            0x00, // ring 2: 1 copy, gap=0, distance=(4-2)=2 backwards ((2<<1)|1 = 5), n-2=0 (n=2)
            0x03,
            0x02,
            0x00,
            0x00, // Start: ring 1 start (0, 0)
            0x01,
            0x00, // Point
            0x07,
            0x06,
            0x14,
            0x00,
            0x00,
            0x14,
            0x14,
            0x00, // Ring: (+10,0), (0,+10), (+10,0) for ring2[2]
            0x05,
            0x04,
            0x00,
            0x00,
            0x00,
            0x00, // Corner: ring1[3]=(0,10), ring2[3]=v[0]+v[2]-v[1]=(10,10)+(20,0)-(10,0)=(20,10)
            0x01,
            0x00, // Line1
            0x01,
            0x00, // Line
        ]);
        const types = new Uint8Array([GEOM_POLYGON, GEOM_POLYGON]);
        const levels = {
            partOffsets: new Uint32Array([0, 1, 2]),
            ringOffsets: new Uint32Array([0, 4, 8]),
        };
        const decoded = decodePredictedVertices(payload, types, levels, 8);
        expect(Array.from(decoded)).toEqual([0, 0, 10, 0, 10, 10, 0, 10, 10, 10, 10, 0, 20, 0, 20, 10]);

        // Escaped multi-byte residuals reaching i32::MAX and i32::MIN on a 2-point layer without topology
        const maxZ = BigInt(0x7fffffff) * 2n - 255n;
        const minDelta = BigInt(-0x80000000) - BigInt(0x7fffffff); // -4294967295
        const minZ = (-minDelta << 1n) - 1n - 255n;
        const encVarint = (val: bigint): number[] => {
            const bytes: number[] = [];
            let v = val;
            while (v >= 128n) {
                bytes.push(Number(v & 127n) | 128);
                v >>= 7n;
            }
            bytes.push(Number(v));
            return bytes;
        };
        const hiBytes = [...encVarint(maxZ), 0x00, ...encVarint(minZ), 0x00];
        const pointBlock = [0x04, 0xff, 0xff, 0xff, 0xff, ...hiBytes];
        const extremePayload = new Uint8Array([
            0x00,
            0x01,
            0x00,
            pointBlock.length,
            ...pointBlock,
            0x01,
            0x00,
            0x01,
            0x00,
            0x01,
            0x00,
            0x01,
            0x00,
        ]);
        const ptTypes = new Uint8Array([GEOM_POINT, GEOM_POINT]);
        const pts = decodePredictedVertices(extremePayload, ptTypes, {}, 2);
        expect(Array.from(pts)).toEqual([2147483647, -128, -2147483648, -256]);

        // Out-of-i32 vertex coordinate (i32::MAX + 1) must be rejected
        const overZ = (BigInt(0x7fffffff) + 1n) * 2n - 255n;
        const overBlock = [0x02, 0xff, 0x00, ...encVarint(overZ)];
        const overPayload = new Uint8Array([
            0x00,
            0x01,
            0x00,
            overBlock.length,
            ...overBlock,
            0x01,
            0x00,
            0x01,
            0x00,
            0x01,
            0x00,
            0x01,
            0x00,
        ]);
        expect(() => decodePredictedVertices(overPayload, new Uint8Array([GEOM_POINT]), {}, 1)).toThrow(RangeError);

        // FLAG_FROM_END (0x04): second line segment predicts its start from previous end (5, 7) instead of previous start (1, 2)
        const fromEndPayload = new Uint8Array([
            0x04, // FLAG_FROM_END
            0x05,
            0x04,
            0x02,
            0x04,
            0x00,
            0x00, // Start: seg1=(1,2), seg2=(5,7)+(0,0)=(5,7)
            0x01,
            0x00,
            0x01,
            0x00,
            0x01,
            0x00,
            0x05,
            0x04,
            0x08,
            0x0a,
            0x02,
            0x06, // Line1: seg1[1]=(1+4,2+5)=(5,7), seg2[1]=(5+1,7+3)=(6,10)
            0x01,
            0x00,
        ]);
        const chainedRuns = { lengths: new Uint32Array([2, 2]), kinds: new Uint8Array([RUN_LINE, RUN_LINE]) };
        expect(Array.from(decodePredictedRuns(fromEndPayload, chainedRuns, 4))).toEqual([1, 2, 5, 7, 5, 7, 6, 10]);

        // Non-canonical trailing zero in escaped varint must be rejected
        const badHi = [0x02, 0xff, 0xff, 0x80, 0x00, 0x00];
        const badPayload = new Uint8Array([
            0x00,
            0x01,
            0x00,
            badHi.length,
            ...badHi,
            0x01,
            0x00,
            0x01,
            0x00,
            0x01,
            0x00,
            0x01,
            0x00,
        ]);
        expect(() => decodePredictedVertices(badPayload, new Uint8Array([GEOM_POINT]), {}, 1)).toThrow();
        expect(
            predictedRuns(new Uint8Array([GEOM_LINESTRING]), { partOffsets: new Uint32Array([0, 2]) }, 2).kinds[0],
        ).toBe(RUN_LINE);
    });

    it("decodes all real-world test/expected/0x02 tiles consistently across columnar and FeatureTable decoders", () => {
        const utf8Enc = new TextEncoder();
        const normPbfStrings = (fc: GeoJSON.FeatureCollection): GeoJSON.FeatureCollection => {
            for (const f of fc.features) {
                if (!f.properties) continue;
                for (const [k, v] of Object.entries(f.properties)) {
                    if (typeof v === "string" && v.charCodeAt(0) === 0xfeff && utf8Enc.encode(v).length >= 12) {
                        f.properties[k] = v.slice(1);
                    }
                }
            }
            return fc;
        };
        const suites = readdirSync(EXPECTED_02, { withFileTypes: true })
            .filter((d) => d.isDirectory())
            .map((d) => d.name)
            .sort();
        let tileCount = 0;
        for (const suite of suites) {
            const suiteDir = path.join(EXPECTED_02, suite);
            const files = readdirSync(suiteDir)
                .filter((f) => f.endsWith(".mlt"))
                .sort();
            for (const file of files) {
                const bytes = new Uint8Array(readFileSync(path.join(suiteDir, file)));
                const colTile = decodeTileColumns(bytes);
                const tables = decodeTile(bytes, undefined, false);
                expect(tables.length).toBe(colTile.layers.length);
                expect(featureTablesToFeatureCollection(tables), `${suite}/${file}`).toEqual(
                    normPbfStrings(tileGeoJson(bytes)),
                );
                tileCount++;
            }
        }
        expect(tileCount).toBeGreaterThanOrEqual(160);
    }, 30_000);
});
