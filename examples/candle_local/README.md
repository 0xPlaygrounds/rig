# Local rig-candle example

This workspace example downloads three Hugging Face files and passes their owned
bytes through the root `rig` facade to `rig-candle`. All filesystem and network
access stays in the example application; `rig-candle` only receives `ModelData`.

From the repository root:

```bash
./examples/candle_local/download_model.sh
cargo run -p candle_local -- "Explain ownership in one sentence."
```

The default model is
[`yujiepan/llama-3-tiny-random`](https://huggingface.co/yujiepan/llama-3-tiny-random).
It is an ungated, single-checkpoint Llama 3 test model whose weights are only
about 2 MB. Its output is random and is useful only for verifying download,
loading, prompt formatting, and inference.
The example prints generated fragments as they arrive, then reports final usage,
finish reason, context limits, prefill time, time to first token, total duration,
and throughput.

To try a useful public instruct model instead, download the approximately 2.5 GB
Llama 3.2 1B checkpoint:

```bash
MODEL_REPO=unsloth/Llama-3.2-1B-Instruct \
  ./examples/candle_local/download_model.sh
cargo run -p candle_local -- "Explain ownership in one sentence."
```

Set `MODEL_DIR` for both commands to store and load the three files elsewhere.
Delete or move an existing model directory before changing `MODEL_REPO`, because
the downloader preserves complete files that are already present.
