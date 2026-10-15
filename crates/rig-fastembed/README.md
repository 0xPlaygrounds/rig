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

## Image embeddings

`FastembedImage` embeds encoded image files (PNG, JPEG, ...) with a `fastembed`
image model, such as CLIP's vision encoder. Give it to an `EmbeddingsBuilder`
with `image_model`, and the images of each `Document` (or any `Embed` type that
calls `Embedder::image`) are embedded beside its text. The CLIP text and vision
models share one space, so a text query finds images:

```rust
use rig_core::completion::Document;
use rig_core::embeddings::EmbeddingsBuilder;
use rig_core::message::Image;
use rig_fastembed::{Fastembed, FastembedImage, FastembedImageModel, FastembedModel};

let text = Fastembed::load(&FastembedModel::ClipVitB32)?
    .embedding(&FastembedModel::ClipVitB32, None)?;
let images = FastembedImage::load(&FastembedImageModel::ClipVitB32)?
    .embedding(&FastembedImageModel::ClipVitB32, None)?;
let photo = Document::new("cat.png", "").with_image(Image::from_bytes(&std::fs::read("cat.png")?));
let embeddings = EmbeddingsBuilder::new(text)
    .image_model(images)
    .document(photo)?
    .build()
    .await?;
```

The workspace example [`image_rag`](../../examples/image_rag/src/main.rs) indexes
images this way and sends the one a question retrieves to a vision model.
