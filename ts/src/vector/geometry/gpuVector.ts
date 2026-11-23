import Point from "@mapbox/point-geometry";
import { createFlatGeometryVector } from "./flatGeometryVector";
import type { CoordinatesArray } from "./geometryVector";
import type { TopologyVector } from "./topologyVector";

export abstract class GpuVector implements Iterable<CoordinatesArray> {
    protected constructor(
        private readonly _triangleOffsets: Uint32Array,
        private readonly _indexBuffer: Uint32Array,
        private readonly _vertexBuffer: Int32Array | Uint32Array,
        private readonly _topologyVector?: TopologyVector,
    ) {}

    abstract geometryType(index: number): number;

    abstract get numGeometries(): number;

    abstract containsSingleGeometryType(): boolean;

    get triangleOffsets(): Uint32Array {
        return this._triangleOffsets;
    }

    get indexBuffer(): Uint32Array {
        return this._indexBuffer;
    }

    get vertexBuffer(): Int32Array | Uint32Array {
        return this._vertexBuffer;
    }

    get topologyVector(): TopologyVector | undefined {
        return this._topologyVector;
    }

    get isTrianglesOnly(): boolean {
        return !this._topologyVector;
    }

    getGeometries(): CoordinatesArray[] {
        if (!this._topologyVector) {
            const vb = this._vertexBuffer;
            const ib = this._indexBuffer;
            const triOff = this._triangleOffsets;
            const geometries: CoordinatesArray[] = new Array(this.numGeometries);
            for (let i = 0; i < this.numGeometries; i++) {
                const t0 = triOff[i];
                const t1 = triOff[i + 1];
                const rings: CoordinatesArray = new Array(t1 - t0);
                for (let t = t0; t < t1; t++) {
                    const a = ib[t * 3] * 2;
                    const b = ib[t * 3 + 1] * 2;
                    const c = ib[t * 3 + 2] * 2;
                    const pa = new Point(vb[a], vb[a + 1]);
                    const pb = new Point(vb[b], vb[b + 1]);
                    const pc = new Point(vb[c], vb[c + 1]);
                    rings[t - t0] = [pa, pb, pc, pa];
                }
                geometries[i] = rings;
            }
            return geometries;
        }
        const types = new Uint32Array(this.numGeometries);
        for (let i = 0; i < this.numGeometries; i++) {
            types[i] = this.geometryType(i);
        }
        return createFlatGeometryVector(types, this._topologyVector, undefined, this._vertexBuffer).getGeometries();
    }

    [Symbol.iterator](): Iterator<CoordinatesArray> {
        return null;
    }
}
