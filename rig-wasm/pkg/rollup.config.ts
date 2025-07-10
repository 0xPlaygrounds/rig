// rollup.config.ts
import type { RollupOptions } from "rollup";
import typescript from "@rollup/plugin-typescript";
import copy from "rollup-plugin-copy";
import wasm from "@rollup/plugin-wasm";

const config: RollupOptions[] = [
  // ESM build
  {
    input: {
      index: "index.ts",
      openai: "openai.ts",
    },
    output: {
      dir: "out/esm",
      format: "esm",
      preserveModules: true, // Preserve module structure for ESM
    },
    plugins: [
      typescript({
        // For ESM, generate declarations directly into 'out/esm' alongside the JS files
        declaration: true,
        declarationDir: "out/esm",
      }),
      wasm(),
    ],
  },
  // CJS build
  {
    input: "index.ts",
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
      }),
      wasm(),
      copy({
        targets: [
          {
            src: "generated/rig_wasm_bg.wasm",
            dest: "out/esm/generated",
          },
          {
            src: "generated/rig_wasm_bg.wasm.d.ts",
            dest: "out/esm/generated",
          },
        ],
      }),
    ],
  },
];

export default config;
