// rollup.config.ts
import type { RollupOptions } from "rollup";
import typescript from "@rollup/plugin-typescript";
import dts from "rollup-plugin-dts";
import wasm from "@rollup/plugin-wasm";
import fs from "fs";
import path from "path";

const providersDir = "src/providers";
const vectorStoresDir = "src/vector_stores";
const coreDir = "src";

const readDir = (inputDir: string): string[] => {
  const files = fs
    .readdirSync(inputDir)
    .filter((file: string) => file.endsWith(".ts")) // or any other filter you need
    .map((file: string) => path.join(inputDir, file));

  return files;
};

const input: Record<string, string> = {};

const dirs = [providersDir, vectorStoresDir, coreDir];

for (const dir of dirs) {
  for (const file of readDir(dir)) {
    const name = path.basename(file, ".ts");
    input[name] = file;
  }
}

const config: RollupOptions[] = [
  // ESM build
  {
    external: [
      "@qdrant/js-client-rest",
      "node:fs",
      "node:path",
      "node:process",
      "util",
    ],
    input: input,
    output: {
      dir: "dist/esm",
      format: "esm",
      preserveModules: true, // Preserve module structure for ESM
    },
    plugins: [
      typescript({
        declaration: true,
        declarationDir: "dist/esm",
        moduleResolution: "node",
        // Skip lib checking for external deps
        skipLibCheck: true,
      }),
      wasm(),
    ],
    onwarn(warning, warn) {
      if (warning.code === "EMPTY_CHUNK") {
        return;
      }
    },
  },
  // CJS build
  {
    external: [
      "@qdrant/js-client-rest",
      "node:fs",
      "node:path",
      "node:process",
      "util",
    ],
    input: "./src/index.ts",
    output: {
      file: "dist/cjs/index.cjs",
      format: "cjs",
    },
    plugins: [
      typescript({
        // Do not generate declarations here.
        // The ESM build already handles declaration generation into 'out/esm',
        // which can be used by both ESM and CJS consumers.
        // The 'outDir' is overridden to ensure compiled JS goes to 'out' before Rollup moves it.
        outDir: "dist",
        moduleResolution: "node",
        // Skip lib checking for external deps
        skipLibCheck: true,
      }),
      wasm(),
    ],
  },
  {
    input: "./src/index.ts", // Use your main entry point for declarations
    output: {
      file: "dist/esm/index.d.ts", // Output the bundled declarations to your ESM types location
      format: "esm", // Format doesn't strictly matter for declarations, but esm is common
    },
    plugins: [
      dts(), // Use the dts plugin to bundle all declarations
    ],

    onwarn(warning, warn) {
      if (warning.code === "EMPTY_CHUNK") {
        return;
      }
    },
  },
];

export default config;
