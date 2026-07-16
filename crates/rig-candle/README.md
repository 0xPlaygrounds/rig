# rig-candle

`rig-candle` runs validated local checkpoints through Rig's `CompletionModel`
and agent APIs. The crate receives byte buffers and performs no filesystem or
network access itself.

```rust,no_run
use rig_core::{agent::AgentBuilder, completion::Prompt};
use rig_candle::{CandleModel, ModelData};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = CandleModel::from_safetensors(ModelData {
        config: std::fs::read("./model/config.json")?,
        tokenizer: std::fs::read("./model/tokenizer.json")?,
        weights: std::fs::read("./model/model.safetensors")?,
    })?;
    let agent = AgentBuilder::new(model)
        .preamble("You are a concise assistant.")
        .build();
    println!("{}", agent.prompt("Explain ownership briefly.").await?);
    Ok(())
}
```

## Validated model profiles

- Llama 3 instruct: one unsharded safetensors checkpoint, explicit Llama 3
  conversation formatting, text conversations.
- SmolLM2-360M-Instruct: Q4_K_M GGUF using Candle's quantized Llama backend,
  explicit ChatML formatting. This remains the small native/WASM demo model.
- Qwen3-4B-Instruct: the official `Qwen3-4B-Q4_K_M.gguf`, using Candle's
  quantized Qwen3 backend and the explicit Qwen Hermes tool protocol. This is
  the native-only agent-conformance profile; it is rejected on wasm32 because
  its runtime memory cannot fit reliably in wasm32 linear memory.

`ModelArtifacts` selects safetensors versus GGUF explicitly. `from_gguf_bytes`
accepts long-lived borrowed bytes such as `include_bytes!` without first copying
the checkpoint. Arbitrary Qwen, Qwen2, Qwen3 MoE/vision, other sizes, shards,
and unvalidated quantizations are rejected rather than treated as Llama.

The loader validates architecture metadata before tensor allocation, exact
profile dimensions and special-token agreement, the complete GGUF vocabulary,
required tensor names/shapes, and the allowed Q4_K_M tensor mix. The effective
GGUF context limit is currently 4096 tokens because that is Candle 0.11's
quantized cache capacity.

## Qwen3 tools and output behavior

Qwen3 supports function name, description, and arbitrary object-shaped JSON
Schema parameters, including optional properties, enums, nested objects, and
arrays. `Auto`, `None`, `Required`, and `Specific` tool choices are represented
explicitly. `Specific` exposes only the named functions; `Required` and
`Specific` add a mandatory-call instruction and reject a model turn with no
call. Unknown functions, malformed/non-object arguments, duplicate IDs,
truncated envelopes, and unmatched results are typed errors.

Argument validation is intentionally split: `rig-candle` validates the JSON
envelope and requires arguments to be an object, while Rig's registered tool
performs typed deserialization/schema validation before execution. Tool
execution always remains in Rig's agent driver. Assistant tool calls and
correlated text or JSON results round-trip through history. Multimodal tool
results and provider-hosted tools are rejected.

Qwen tool syntax can cross token boundaries, so streaming buffers one model
turn, parses it, then emits ordered text/reasoning items and complete
`RawStreamingChoice::ToolCall` values followed by `FinalResponse`. It does not
currently emit tool-call deltas. This keeps model XML out of user-visible text
while preserving cancellation, bounded backpressure, and the same parsed result
as buffered completion.

The explicit prompt requests Qwen's no-thinking mode. If a checkpoint emits a
leading `<think>` block anyway, it is represented as Rig reasoning content and
omitted from later rendered history; control syntax is never exposed as normal
text.

Direct `CompletionRequest::output_schema` returns a typed unsupported-feature
error: decoding is not grammar constrained. Rig's `OutputMode::Tool` works and
is the supported path for a real tool followed by a synthetic structured final
result. That mode is best-effort and Rig validates the returned JSON; it is not
native grammar enforcement.

## Pinned live model

Download the official artifacts with the reproducible helper:

```bash
export RIG_CANDLE_TEST_MODEL_DIR="$PWD/crates/rig-candle/test-models/qwen3-4b-q4-k-m"
./crates/rig-candle/tests/download_qwen3.sh
cargo test -p rig-candle --test live_conformance \
  -- --ignored --nocapture --test-threads=1
```

The script uses immutable revisions, retries resumable temporary downloads,
checks size and SHA-256, then atomically installs each file. Verified files are
reused and `test-models/` is ignored.

| artifact | revision | bytes | SHA-256 |
|---|---|---:|---|
| `Qwen3-4B-Q4_K_M.gguf` | `bc640142c66e1fdd12af0bd68f40445458f3869b` | 2,497,280,256 (2.33 GiB) | `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5` |
| `tokenizer.json` | `1cfa9a7208912126459214e8b04321603b3df60c` | 11,422,654 (10.89 MiB) | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| `config.json` | `1cfa9a7208912126459214e8b04321603b3df60c` | 726 | `8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a` |

The live test loads the model once, reuses cheap `CandleModel` clones with a
fresh KV cache per request, applies greedy decoding and a fixed seed, enforces
timeouts, and prints tokens, tool counts, history size, timings, throughput, and
safe output. On the ARM64 development host used for the release-mode verification,
the seven-scenario suite completed in 72.45 seconds: the short direct completion
reported 2.03 generated tokens/s and the multi-turn reports ranged from 4.07 to
5.90 aggregate generated tokens/s per end-to-end second. Actual resident memory
and speed depend on the target; the loaded quantized tensors are based on a
2.33-GiB GGUF, KV cache grows with context, and loading temporarily also holds
the 2.33-GiB input byte buffer alongside constructed tensors. Plan for more than
twice the checkpoint size during loading rather than treating file size as a
hard memory bound.

## Conformance boundary

Portable model-contract helpers live under `rig_core::test_utils` and are shared
by the Ollama Qwen3 cassette suite and this artifact-backed Candle suite. They
cover optional arguments, sequential tools, streaming execution, history, and a
synthetic structured final result on both buffered and streaming surfaces, plus
all portable tool-choice modes. Provider suites retain HTTP serialization,
authentication, SSE parsing, hosted tools, remote files, and provider-specific
reasoning/session assertions; those are transport conformance, not local model
quality.

## Runtime behavior and limits

Builder defaults can be overridden by request `max_tokens` and `temperature`.
The Candle-only `additional_params` keys are `top_k`, `top_p`, `seed`,
`repeat_penalty`, and `repeat_last_n`; unknown keys are errors. Native inference
runs in `spawn_blocking`; `max_concurrent_requests` defaults to one. Every
request owns its cache and sampler. Cancellation is cooperative between Candle
forwards, and the bounded stream channel has capacity eight.

WASM inference is synchronous and should run in an application-owned Web Worker.
The maintained browser example embeds SmolLM2 at compile time. CPU only; no
CUDA/Metal selection, batching, multimodal prompts, or checkpoint shards.
