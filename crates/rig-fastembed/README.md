## Fastembed integration with Rig
This crate provides local [`fastembed-rs`](https://github.com/Anush008/fastembed-rs) embedding models for Rig — implementations of `rig_core::embeddings::EmbeddingModel` that run on-device (no API key).

Unlike the providers found in the core crate, `fastembed` does not compile to the `wasm32-unknown-unknown` target.

## Installation

```toml
[dependencies]
rig-fastembed = "0.4.0"
rig-core = "0.36.0"
```

The default features enable Hugging Face model downloads and ONNX Runtime binary
downloads through `fastembed`. The root `rig` facade exposes this crate with the
`fastembed`, `fastembed-hf-hub`, and `fastembed-ort-download-binaries` features.

## Retrieval (RAG)

Rig does not ship a built-in vector-store abstraction. Generate embeddings with
this crate's models, then own storage and search in user-land and surface
retrieval to the agent as either:

- an ordinary `Tool` the model calls (active RAG), or
- an `AgentHook` that injects retrieved context via `RequestPatch::extra_context`
  before each model call (passive RAG).

See the root workspace examples `tool_active_rag` and `hook_passive_rag`.
