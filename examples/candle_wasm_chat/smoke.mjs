import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

import init, {
    chat,
    clear_history,
    embedded_model_size,
    history_len,
    initialize,
    model_is_embedded,
} from "./www/pkg/candle_wasm_chat.js";

const wasm = await readFile(new URL("./www/pkg/candle_wasm_chat_bg.wasm", import.meta.url));
await init({ module_or_path: wasm });
assert.equal(model_is_embedded(), true);
assert.equal(embedded_model_size(), 272_696_282);

initialize();
initialize();
assert.equal(history_len(), 0);

const output = await chat("Reply with one short greeting.");
assert.ok(output.trim().length > 0);
assert.ok(history_len() >= 2);

clear_history();
assert.equal(history_len(), 0);
console.log(`WASM smoke passed: ${JSON.stringify(output)}`);
