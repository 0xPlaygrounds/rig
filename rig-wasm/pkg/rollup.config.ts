// rollup.config.ts
import type { RollupOptions } from "rollup";
import typescript from "@rollup/plugin-typescript";
import dts from "rollup-plugin-dts";
import wasm from "@rollup/plugin-wasm";

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
    input: {
      index: "./src/index.ts",
      openai: "./src/providers/openai.ts",
      gemini: "./src/providers/gemini.ts",
      anthropic: "./src/providers/anthropic.ts",
      types: "./src/types.ts",
      qdrant: "./src/vector_stores/qdrant.ts",
      vector_store: "./src/vector_stores/vector_store.ts",
      streaming: "./src/streaming.ts",
      utils: "./src/utils.ts",
    },
    output: {
      dir: "out/esm",
      format: "esm",
      preserveModules: true, // Preserve module structure for ESM
    },
    plugins: [
      typescript({
        declaration: true,
        declarationDir: "out/esm",
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
      file: "out/cjs/index.cjs",
      format: "cjs",
    },
    plugins: [
      typescript({
        // Do not generate declarations here.
        // The ESM build already handles declaration generation into 'out/esm',
        // which can be used by both ESM and CJS consumers.
        // The 'outDir' is overridden to ensure compiled JS goes to 'out' before Rollup moves it.
        outDir: "out",
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
      file: "out/esm/index.d.ts", // Output the bundled declarations to your ESM types location
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
