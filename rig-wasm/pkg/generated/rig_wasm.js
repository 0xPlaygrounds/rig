
let imports = {};
import * as import0 from './rig_wasm_bg.js';
imports['./rig_wasm_bg.js'] = import0;

import * as path from 'node:path';
import * as fs from 'node:fs';
import * as process from 'node:process';

let file = path.dirname(new URL(import.meta.url).pathname);
if (process.platform === 'win32') {
    file = file.substring(1);
}
const bytes = fs.readFileSync(path.join(file, 'rig_wasm_bg.wasm'));

const wasmModule = new WebAssembly.Module(bytes);
const wasmInstance = new WebAssembly.Instance(wasmModule, imports);
const wasm = wasmInstance.exports;
export const __wasm = wasm;

imports["./rig_wasm_bg.js"].__wbg_set_wasm(wasm, wasmModule);
wasm.__wbindgen_start();

export * from "./rig_wasm_bg.js";