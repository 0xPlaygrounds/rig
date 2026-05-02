## Fastembed integration with Rig
This crate allows you to use [`fastembed-rs`](https://github.com/Anush008/fastembed-rs) with Rig.

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

See [`examples/vector_search_fastembed.rs`](./examples/vector_search_fastembed.rs)
and [`examples/vector_search_fastembed_local.rs`](./examples/vector_search_fastembed_local.rs)
for end-to-end vector search examples.
