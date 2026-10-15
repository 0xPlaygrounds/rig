## Fastembed integration with Rig
This crate provides local text and image embeddings through
[`fastembed-rs`](https://github.com/Anush008/fastembed-rs).

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

## Image embeddings

Load an image model with `FastembedImage` and pass encoded image bytes to
`Model::call`. The first load downloads the model; inference runs locally.

```rust,no_run
use rig_fastembed::{FastembedImage, FastembedImageModel};

# async fn run() -> anyhow::Result<()> {
let runtime = FastembedImage::load(&FastembedImageModel::ClipVitB32)?;
let model = runtime.embedding(&FastembedImageModel::ClipVitB32, None);
let images = vec![std::fs::read("photo.png")?];
let response = model.call(images).await?;
for embedding in response.embeddings {
    println!("{}: {} dimensions", embedding.document, embedding.vec.len());
}
# Ok(())
# }
```

The response contains one vector per image, in input order. Each vector's
`document` is a media type and SHA-256 identifier for the original image bytes.
Invalid images fail the batch. A supplied dimension count validates the model's
output width; it does not resize vectors.

`FastembedImage::from_user_defined` loads a caller-supplied ONNX model and
preprocessor configuration without Hugging Face access. Pair that runtime with
a `rig_core::driver::Local<rig_core::operation::ImageEmbedding>` wire describing
its model ID and dimensions. The `image_embeddings` factory supplies this wire
for named FastEmbed models.

Run the [image embedding example](./examples/image_embeddings_fastembed.rs) with:

```sh
cargo run -p rig-fastembed --example image_embeddings_fastembed -- photo.png another.jpg
```
