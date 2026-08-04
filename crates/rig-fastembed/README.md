## Fastembed integration with Rig
This crate allows you to use [`fastembed-rs`](https://github.com/Anush008/fastembed-rs) with Rig.

Unlike the providers found in the core crate, `fastembed` does not compile to the `wasm32-unknown-unknown` target.

## Installation

```toml
[dependencies]
rig-fastembed = "0.4.0"
rig-core = "0.36.0"
```

## Public shape

There is no client type and no model trait. The crate is a config record, a
loaded handle, and free functions:

```rust,no_run
use rig_core::embeddings::embed_documents;
use rig_fastembed::{EmbeddingConfig, FastembedModel, MAX_DOCUMENTS, functions};

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
// `EmbeddingConfig` says which weights to load; `load()` returns the handle.
let model = EmbeddingConfig::new(FastembedModel::AllMiniLML6V2Q).load()?;

// One-shot texts.
let response = functions::embed(&model, vec!["Hello, world!".to_string()])?;

// Whole `#[embed]` documents — the replacement for `EmbeddingsBuilder`.
let documents = embed_documents(my_docs, MAX_DOCUMENTS, 1, |texts| async {
    functions::embed(&model, texts)
})
.await?;
# let _ = (response, documents);
# Ok(())
# }
```

`functions::embed_text` embeds a single string (handy for queries),
`functions::embed_batches` embeds caller-defined batches, and
`functions::DESCRIPTOR` is the capability sheet. Local weights load through
`EmbeddingModel::new_from_user_defined`. Because FastEmbed inference is a local
ONNX session, `embed` is synchronous and reports zero token usage.

The default features enable Hugging Face model downloads and ONNX Runtime binary
downloads through `fastembed`. The root `rig` facade exposes this crate with the
`fastembed`, `fastembed-hf-hub`, and `fastembed-ort-download-binaries` features.

See [`examples/vector_search_fastembed.rs`](./examples/vector_search_fastembed.rs)
and [`examples/vector_search_fastembed_local.rs`](./examples/vector_search_fastembed_local.rs)
for end-to-end vector search examples.
