#!/usr/bin/env bash
./scripts/exports.sh && \
npm ci && \
rollup -c && \
cp src/generated/rig_wasm_bg.wasm out/esm/generated && \
cp src/generated/rig_wasm_bg.wasm.d.ts out/esm/generated
