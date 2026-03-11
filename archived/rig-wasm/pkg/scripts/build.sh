#!/usr/bin/env bash
./scripts/exports.sh && \
npm ci && \
rollup -c && \
cp src/generated/rig_wasm_bg.wasm dist/esm/generated && \
cp src/generated/rig_wasm_bg.wasm.d.ts dist/esm/generated
