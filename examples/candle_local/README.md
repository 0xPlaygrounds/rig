# Local rig-candle example

This workspace example downloads three pinned Hugging Face files and passes their owned
bytes through the root `rig` facade to `rig-candle`. All filesystem and network
access stays in the example application; `rig-candle` only receives `ModelData`.

From the repository root:

```bash
./examples/candle_local/download_model.sh
cargo run -p candle_local -- "Explain ownership in one sentence."
```

The default model is SmolLM2-360M-Instruct in Q4_K_M GGUF form. Its
270,590,880-byte checkpoint produces useful instruction responses while staying
small enough for the browser example.
The example prints generated fragments as they arrive, then reports final usage,
finish reason, context limits, prefill time, time to first token, total duration,
and throughput.

Set `MODEL_DIR` for both commands to store and load the three files elsewhere.
The downloader verifies immutable revisions and SHA-256 checksums and preserves
only already complete files.
