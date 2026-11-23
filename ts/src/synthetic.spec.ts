import { readFile } from "node:fs/promises";
import * as path from "node:path";
import { describe, expect, it } from "vitest";
import { featureTablesToFeatureCollection } from "./vector/featureTablesToGeoJson";
import {
    compareWithTolerance,
    expectUnsupported,
    getTestCases,
    writeActualOutput,
} from "../../test/synthetic/synthetic-test-utils";
import decodeTile from "./mltDecoder";

/**
 * Synthetics the decoder cannot handle yet. These still run: `expectUnsupported` asserts they fail,
 * so an entry that starts decoding correctly fails the test until it is removed from this list.
 * Prefer fixing the decoder over adding to it.
 */
const UNIMPLEMENTED_SYNTHETICS: string[] = [
    // v2 3D Meshes
    "0x02/mesh_mvalues_both",
    "0x02/mesh_mvalues_triangle",
    "0x02/mesh_mvalues_vertex",
    "0x02/mesh_relative_to_ground",
    "0x02/mesh_roof_wall",
    "0x02/mesh_triangle",
    // v2 M-value columns
    "0x02/mvalues",
    "0x02/mvalues_15props_7cols",
    "0x02/mvalues_16props",
    "0x02/mvalues_8cols",
    "0x02/mvalues_all_null",
    "0x02/mvalues_bool",
    "0x02/mvalues_delta_rle",
    "0x02/mvalues_f32",
    "0x02/mvalues_f32_dict",
    "0x02/mvalues_f64",
    "0x02/mvalues_f64_alp",
    "0x02/mvalues_fpf",
    "0x02/mvalues_i32",
    "0x02/mvalues_i64",
    "0x02/mvalues_i8",
    "0x02/mvalues_line_dict",
    "0x02/mvalues_mline",
    "0x02/mvalues_mline_dict",
    "0x02/mvalues_mpoint",
    "0x02/mvalues_mpoint_dict",
    "0x02/mvalues_mpoly",
    "0x02/mvalues_mpoly_dict",
    "0x02/mvalues_plain",
    "0x02/mvalues_poly",
    "0x02/mvalues_poly_dict",
    "0x02/mvalues_poly_hole",
    "0x02/mvalues_rle",
    "0x02/mvalues_sp",
    "0x02/mvalues_sp_prop",
    "0x02/mvalues_str_dict",
    "0x02/mvalues_str_fsst",
    "0x02/mvalues_str_plain",
    "0x02/mvalues_tes",
    "0x02/mvalues_u32",
    "0x02/mvalues_u64",
    "0x02/mvalues_u8",
    // v2 Nested columns
    "0x02/nested_list_struct",
    "0x02/nested_map_shapes",
    "0x02/nested_map_str",
    "0x02/nested_struct",
    "0x02/nested_struct_shapes",
    // v2 3D Z coordinates
    "0x02/z_height_absolute",
    "0x02/z_height_relative_to_ground",
    "0x02/z_line",
    "0x02/z_line_fpf",
    "0x02/z_mix",
    "0x02/z_mvalues",
    "0x02/z_point",
    "0x02/z_points",
    "0x02/z_points_hilbert",
    "0x02/z_poly_hole",
    "0x02/z_poly_hole_tes",
    "0x02/z_poly_hole_tri",
    "0x02/z_step_coarsest",
    "0x02/z_step_finest",
];

describe("MLT Decoder - Synthetic tests", () => {
    expect.addEqualityTesters([compareWithTolerance]);
    const testCases = getTestCases(UNIMPLEMENTED_SYNTHETICS.map((entry) => entry.replaceAll("/", path.sep)));
    for (const { name, content, fileName } of testCases.active) {
        it(name, async () => {
            const actual = await decodeMLT(fileName);
            writeActualOutput(fileName, actual);
            expect(actual).toEqual(content);
        });
    }

    for (const { name, content, fileName } of testCases.skipped) {
        it(`${name} (unsupported)`, () => expectUnsupported(() => decodeMLT(fileName), content));
    }
});

async function decodeMLT(mltFilePath: string) {
    const mltBuffer = await readFile(mltFilePath);
    const featureTables = decodeTile(mltBuffer, undefined, false);
    return featureTablesToFeatureCollection(featureTables) as unknown as Record<string, unknown>;
}
