# rig-candle

`rig-candle` adapts byte-backed Llama-family instruct models to Rig's
`CompletionModel` and agent APIs. It supports unsharded safetensors and Q4_K_M
GGUF on native targets and `wasm32-unknown-unknown`.

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

For quantized loading, pass the same config and tokenizer plus `model.gguf` to
`LlamaModel::from_gguf`. Embedded applications can use `from_gguf_bytes` and
`GgufModelData` to borrow `include_bytes!` buffers without copying the checkpoint.
`ModelArtifacts` makes the format explicit; the crate does not infer it from a
filename. The crate never reads files or uses the network.

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
Llama 3 or SmolLM2 special tokens; and tied versus separate output embeddings. Errors name
the incompatible tensor, token, field, expected value, and actual value where
applicable. GGUF loading additionally validates the SmolLM2 identity, exact GGUF
token vocabulary, BOS/EOS IDs, Q4_K_M metadata, every required tensor shape, the
allowed mixed tensor encodings, and the complete Candle tensor load before it
reports the model ready.

Prompt rendering is explicit and deterministic. Llama 3 and SmolLM2 use their
documented instruct control tokens; arbitrary tokenizer chat templates are not
executed. The maintained Candle quantized-Llama implementation is used for
SmolLM2 because the official model declares `LlamaForCausalLM` and its GGUF uses
the Llama tensor schema. Candle 0.11's quantized implementation has a 4096-token
cache limit, so GGUF context capacity is capped at 4096 even when config permits
more.

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
directory containing config/tokenizer plus either `model.safetensors` or
`model.gguf`:

```bash
RIG_CANDLE_MODEL_DIR=/path/to/model \
  cargo test -p rig-candle --test real_model -- --ignored --nocapture
```

## MVP limitations

- Llama 3 and SmolLM2 instruct prompt formatting only
- unsharded Llama-family safetensors or SmolLM2-360M-Instruct Q4_K_M GGUF
- one checkpoint file; no shards or index
- CPU inference only; no CUDA, Metal, or device selection
- buffered and streaming text conversations without tools
- no structured output, multimodal input, tool calls, or batching
