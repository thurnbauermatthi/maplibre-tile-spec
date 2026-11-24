import { createReadStream } from "node:fs";
import { readdir, stat } from "node:fs/promises";
import { createServer } from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
// Serve from the repository root so `/ts/viewer/meshes.html` can also reach
// `/rust/*.pmtiles` and `/test/fixtures/meshes/tiles` via `?url=` or `?tiles=`.
const ROOT = path.resolve(__dirname, "../..");
const PORT = Number(process.env.PORT ?? 8090);

const MIME_TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".map": "application/json; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".svg": "image/svg+xml",
  ".mlt": "application/octet-stream",
  ".pmtiles": "application/octet-stream",
  ".wasm": "application/wasm",
};

function isInsideRoot(targetPath) {
  const rel = path.relative(ROOT, targetPath);
  return !rel.startsWith("..") && !path.isAbsolute(rel);
}

async function listMltTiles(baseDir) {
  const tiles = [];
  const zEntries = await readdir(baseDir, { withFileTypes: true }).catch(() => []);
  for (const zDir of zEntries) {
    if (!zDir.isDirectory() || !/^\d+$/.test(zDir.name)) continue;
    const xEntries = await readdir(path.join(baseDir, zDir.name), { withFileTypes: true }).catch(() => []);
    for (const xDir of xEntries) {
      if (!xDir.isDirectory() || !/^\d+$/.test(xDir.name)) continue;
      const yEntries = await readdir(path.join(baseDir, zDir.name, xDir.name), { withFileTypes: true }).catch(() => []);
      for (const yFile of yEntries) {
        if (yFile.isFile() && /^\d+\.mlt$/.test(yFile.name)) {
          tiles.push(`${zDir.name}/${xDir.name}/${yFile.name.slice(0, -4)}`);
        }
      }
    }
  }
  return tiles;
}

const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url ?? "/", `http://${req.headers.host ?? "localhost"}`);
    if (
      url.pathname === "/" ||
      url.pathname === "/ts" ||
      url.pathname === "/ts/" ||
      url.pathname === "/ts/viewer" ||
      url.pathname === "/ts/viewer/"
    ) {
      res.writeHead(302, { Location: `/ts/viewer/meshes.html${url.search}` });
      res.end();
      return;
    }

    const decodedPath = decodeURIComponent(url.pathname).replace(/^\/+/, "");
    let filePath = path.resolve(ROOT, decodedPath);
    if (!isInsideRoot(filePath)) {
      res.writeHead(403);
      res.end("Forbidden");
      return;
    }

    let info = await stat(filePath).catch(() => null);

    if (!info && decodedPath.endsWith("/index.json")) {
      const baseDir = path.dirname(filePath);
      if (isInsideRoot(baseDir)) {
        const tiles = await listMltTiles(baseDir);
        if (tiles.length > 0) {
          const body = Buffer.from(JSON.stringify(tiles));
          res.writeHead(200, {
            "Content-Type": MIME_TYPES[".json"],
            "Content-Length": body.length,
          });
          if (req.method !== "HEAD") res.write(body);
          res.end();
          return;
        }
      }
    }

    // Resolve extensionless relative ESM imports emitted by `tsc` in `ts/dist/`.
    if ((!info || info.isDirectory()) && !path.extname(filePath)) {
      const jsPath = `${filePath}.js`;
      const jsInfo = isInsideRoot(jsPath) ? await stat(jsPath).catch(() => null) : null;
      if (jsInfo?.isFile()) {
        filePath = jsPath;
        info = jsInfo;
      } else if (info?.isDirectory()) {
        const indexJs = path.join(filePath, "index.js");
        const idxInfo = isInsideRoot(indexJs) ? await stat(indexJs).catch(() => null) : null;
        if (idxInfo?.isFile()) {
          res.writeHead(302, { Location: `${url.pathname.replace(/\/+$/, "")}/index.js${url.search}` });
          res.end();
          return;
        }
      }
    }

    if (!info?.isFile()) {
      res.writeHead(404);
      res.end("Not found");
      return;
    }

    const ext = path.extname(filePath).toLowerCase();
    const contentType = MIME_TYPES[ext] ?? "application/octet-stream";
    const range = req.headers.range;

    if (range) {
      const match = /^bytes=(\d+)-(\d*)$/.exec(range);
      if (!match) {
        res.writeHead(416, { "Content-Range": `bytes */${info.size}` });
        res.end();
        return;
      }
      const start = Number(match[1]);
      const end = match[2] ? Math.min(Number(match[2]), info.size - 1) : info.size - 1;
      if (start >= info.size || start > end) {
        res.writeHead(416, { "Content-Range": `bytes */${info.size}` });
        res.end();
        return;
      }
      res.writeHead(206, {
        "Content-Type": contentType,
        "Accept-Ranges": "bytes",
        "Content-Range": `bytes ${start}-${end}/${info.size}`,
        "Content-Length": end - start + 1,
      });
      if (req.method === "HEAD") {
        res.end();
        return;
      }
      createReadStream(filePath, { start, end }).pipe(res);
      return;
    }

    res.writeHead(200, {
      "Content-Type": contentType,
      "Accept-Ranges": "bytes",
      "Content-Length": info.size,
    });
    if (req.method === "HEAD") {
      res.end();
      return;
    }
    createReadStream(filePath).pipe(res);
  } catch {
    res.writeHead(404);
    res.end("Not found");
  }
});

server.listen(PORT, "127.0.0.1", () => {
  console.log(`MLT TS viewer running at:`);
  console.log(`  http://localhost:${PORT}/ts/viewer/meshes.html`);
  console.log(
    `  http://localhost:${PORT}/ts/viewer/meshes.html?tiles=/test/fixtures/meshes/tiles`,
  );
  console.log(
    `  http://localhost:${PORT}/ts/viewer/meshes.html?url=/rust/hamburg_demo.pmtiles`,
  );
  console.log(
    `  http://localhost:${PORT}/ts/viewer/meshes.html?url=/rust/hamburg_demo_lod3_40m.pmtiles`,
  );
});
