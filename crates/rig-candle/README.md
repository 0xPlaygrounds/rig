# rig-candle

`rig-candle` adapts a Hugging Face-format, non-quantized Llama-family instruct
model to Rig's `CompletionModel` and agent APIs. It is CPU-only and works from
owned bytes on native targets and `wasm32-unknown-unknown`.

```rust,no_run
use rig_core::{agent::AgentBuilder, completion::Prompt};
use rig_candle::{LlamaModel, ModelData};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The application owns all I/O. rig-candle only receives these bytes.
    let data = ModelData {
        config: std::fs::read("./model/config.json")?,
        tokenizer: std::fs::read("./model/tokenizer.json")?,
        weights: std::fs::read("./model/model.safetensors")?,
    };

    let model = LlamaModel::builder(data)
        .temperature(0.7)
        .max_tokens(256)
        .build()?;
    let agent = AgentBuilder::new(model)
        .preamble("You are a helpful assistant.")
        .build();

    println!("{}", agent.prompt("Explain Rust ownership briefly.").await?);
    Ok(())
}
```

The directory in this example contains exactly `config.json`, `tokenizer.json`,
and one unsharded `model.safetensors` checkpoint. The crate never reads files or
uses the network.

The repository includes a runnable package with a model downloader and metadata
output under `examples/candle_local`:

```bash
./examples/candle_local/download_model.sh
cargo run -p candle_local -- "Explain ownership in one sentence."
```

## Loading and validation

Before Candle loads the model, `rig-candle` validates every expected tensor name,
shape, and dtype (`F32`, `F16`, or `BF16`); model dimensions and attention-head
relationships; tokenizer/config vocabulary agreement; BOS/EOS IDs; required
Llama 3 special tokens; and tied versus separate output embeddings. Errors name
the incompatible tensor, token, field, expected value, and actual value where
applicable. Only one complete, unsharded safetensors buffer is accepted.

## Generation and context limits

Builder defaults apply unless a request supplies `max_tokens` or `temperature`.
The optional Candle-specific `additional_params` object accepts `top_k`, `top_p`,
`seed`, `repeat_penalty`, and `repeat_last_n`; malformed values and unknown keys
are errors. If the requested output would cross the context boundary, it is
clamped to the remaining capacity. A prompt over the limit, or one that exactly
fills it, is rejected. `CandleCompletionResponse` reports requested and effective
limits, usage, finish reason, prefill duration, time to first token, total
generation duration, and throughput. Timings begin immediately before prompt
tensor creation: prefill ends after the initial full-prompt forward pass, time to
first token ends when the first token (including EOS) is sampled, and total
duration ends after incremental decoding is finalized.
Native channel-delivery and backpressure wait time is excluded from total
generation duration and throughput, so a slow consumer does not change inference
timing metadata.

## Streaming

Buffered and streaming completion use the same private token-by-token generation
session, including the same fresh KV cache, seeded sampler, repeat penalty,
context clamp, stop behavior, usage, and errors. Native streams deliver exact
incremental tokenizer fragments through a bounded channel with capacity eight.
This bounds queued text and applies backpressure to the blocking generator when a
consumer is slow. Concatenating all text fragments produces exactly the buffered
text, including for byte-fallback and multi-token Unicode sequences. The final
stream item carries `CandleCompletionResponse` usage, finish, limit, and timing
metadata.

## Native concurrency and cancellation

Native inference acquires an asynchronous admission permit before entering
`spawn_blocking`. `LlamaModelBuilder::max_concurrent_requests` controls the
per-model limit and defaults to one, avoiding unbounded CPU oversubscription and
simultaneous KV-cache allocations. Every request owns a fresh cache and sampler.

Dropping a native completion future or response stream signals cooperative
cancellation. Generation checks that signal between tokens and before follow-up
forwards, but cannot stop a Candle forward already executing. The concurrency
permit remains owned by the blocking worker until it actually exits; native
threads are never forcibly ended. Closing a stream while its producer is waiting
on backpressure is safe and wakes the producer so it can stop.

On WebAssembly, inference remains synchronous inside the completion or stream
future and does not use native threads or synchronization. Streaming events are
therefore collected synchronously before the stream is returned rather than
arriving concurrently. Run the model in an application-owned Web Worker so CPU
inference does not block the browser UI; `rig-candle` does not create or manage
workers.

## Real-model test

The ignored native integration test performs no download. Point it at a local
directory containing the same three files:

```bash
RIG_CANDLE_MODEL_DIR=/path/to/model \
  cargo test -p rig-candle --test real_model -- --ignored --nocapture
```

## MVP limitations

- Llama 3 instruct prompt formatting only
- non-quantized Hugging Face Llama-family checkpoints only
- one safetensors file; no shards or index
- CPU inference only; no CUDA, Metal, or device selection
- buffered and streaming text conversations without tools
- no structured output, multimodal input, tool calls, or batching
