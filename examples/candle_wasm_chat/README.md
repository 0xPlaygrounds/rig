# Local rig-candle WASM chat

This example compiles SmolLM2-360M-Instruct Q4_K_M into the WebAssembly module and
runs Rig and Candle entirely inside a browser Web Worker. Chat messages never
leave the page.

From the repository root:

```bash
./examples/candle_wasm_chat/build.sh
./examples/candle_wasm_chat/serve.sh
```

Then open <http://127.0.0.1:8080>. Set `PORT` when running `serve.sh` to use a
different port.

`build.sh` first runs `download_model.sh`, which downloads `config.json`,
`tokenizer.json`, and `model.gguf` into the ignored `model/` directory.
The Rust build script copies those bytes into Cargo's build output and
`include_bytes!` embeds them in the final WASM module. `wasm-pack` writes the
browser package to the ignored `www/pkg/` directory. Direct builds verify the
pinned size and SHA-256 of every artifact before copying hundreds of megabytes;
a partial or modified model directory fails early with a recovery instruction.

Set `MODEL_DIR` to choose the artifact directory for both downloading and
embedding. The downloader intentionally pins immutable model revisions and
checksums; it does not accept model overrides.

The model configuration and tokenizer are pinned to the official
[`HuggingFaceTB/SmolLM2-360M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)
repository. The 270,590,880-byte checkpoint is Bartowski's Q4_K_M GGUF. Q4_K_M
is used because Candle supports its GGUF tensor kernels on native and WASM and it
is substantially smaller than full-precision weights while retaining useful
instruction output. The three embedded inputs total 272,696,282 bytes.

The Web Worker prevents synchronous WASM inference from freezing the page. The
Rust side retains a rolling window of recent Rig chat history until the
**Clear** button is pressed. User messages are limited to 1,024 UTF-8 bytes and
the retained window is bounded by both message count and serialized size, so a
long-running demo does not inevitably become stuck beyond the model's 4,096-token
context. Reinitialization is idempotent, avoiding a transient second model
allocation in wasm32's constrained linear memory.
Generation is currently returned after completion because synchronous Candle
inference cannot yield incremental fragments to JavaScript on the same worker.
After the page, worker, and WASM package load, inference makes no network
requests.

The worker controller has a model-independent Node regression test:

```bash
node --test examples/candle_wasm_chat/www/worker-runtime.test.mjs
```

After `build.sh`, exercise the actual embedded model, idempotent initialization,
chat history, and Clear behavior with:

```bash
node examples/candle_wasm_chat/smoke.mjs
```

## Measured reference build

On an Apple Silicon development machine, the release build produced a
277,885,159-byte WASM file. A direct JavaScript/WASM smoke test loaded the package
from the included local server and initialized the 272,696,282 embedded artifact
bytes in 496 ms. The required conversation produced `Hello!`, `Paris`, `Green`,
and then recalled `Green`; turn times were 14.5, 20.2, 26.2, and 31.6 seconds as
the full history grew. After Clear, the model no longer recalled green.

The borrowed embedded-artifact loader avoids a transient 270,590,880-byte GGUF
copy. Node observed about 1.76 GB RSS immediately after initialization and about
1.23 GB after the full conversation, with roughly 760 MB classified as
external/WASM memory. Browser engines, hardware, and optimization settings will
vary, so treat these numbers as a practical reference rather than a guarantee.

The equivalent unoptimized native run generated the same answer in 17,536 ms
(0.46 generated tokens/s), including a 13,914 ms prefill. Release builds are
strongly recommended for actual use.
