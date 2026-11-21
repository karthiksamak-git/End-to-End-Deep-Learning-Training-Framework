/**
 * Copy the onnxruntime-web WASM/MJS runtime into `public/ort/`.
 *
 * Vercel serves `public/` as static assets, so the browser fetches the WASM
 * binary from our own origin (no CDN, no cross-origin issues). Runs in
 * `prebuild`, so the assets are always in sync with the installed package.
 */
import { copyFileSync, existsSync, mkdirSync, readdirSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const webRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const distDir = join(webRoot, "node_modules", "onnxruntime-web", "dist");
const outDir = join(webRoot, "public", "ort");

if (!existsSync(distDir)) {
  console.error(`[copy-ort-wasm] onnxruntime-web is not installed (looked in ${distDir})`);
  process.exit(1);
}

mkdirSync(outDir, { recursive: true });

// Only the plain WASM execution provider is used by the playground, so the
// (much larger) jsep/WebGPU variants are skipped deliberately.
const wanted = readdirSync(distDir).filter(
  (file) => (file.endsWith(".wasm") || file.endsWith(".mjs")) && !file.includes("jsep"),
);

let bytes = 0;
for (const file of wanted) {
  copyFileSync(join(distDir, file), join(outDir, file));
  bytes += statSync(join(distDir, file)).size;
}

console.log(
  `[copy-ort-wasm] copied ${wanted.length} files (${(bytes / 1024 / 1024).toFixed(1)} MB) to public/ort`,
);
