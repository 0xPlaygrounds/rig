//! Embeds an image with Cohere Embed v3.
//!
//! Set `COHERE_API_KEY`, then run:
//!
//! ```text
//! cargo run -p cohere_image_embeddings -- path/to/image.png
//! ```

use anyhow::{Context, Result};
use rig::embeddings::ImageEmbeddingModel;
use rig::prelude::*;
use rig::providers::cohere::Cohere;

#[tokio::main]
async fn main() -> Result<()> {
    let path = std::env::args_os()
        .nth(1)
        .context("pass the path to a PNG, JPEG, WebP, or GIF image")?;
    let image = std::fs::read(&path)
        .with_context(|| format!("failed to read image at {}", path.to_string_lossy()))?;

    let cohere = Cohere::from_env()?.bound()?;
    // Embed v3 embeds images with one fixed model at one fixed width, so the
    // image-embedding wire takes neither a model name nor a dimension count.
    let model = cohere.image_embedding("", None);
    let embedding = model.embed_image(&image).await?;

    println!(
        "embedded {} bytes into {} dimensions",
        image.len(),
        embedding.vec.len()
    );

    Ok(())
}
