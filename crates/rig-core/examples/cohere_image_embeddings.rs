//! Embeds an image with Cohere Embed v3.
//!
//! Set `COHERE_API_KEY`, then run:
//!
//! ```text
//! cargo run -p rig-core --example cohere_image_embeddings -- path/to/image.png
//! ```

use anyhow::{Context, Result};
use rig_core::{
    client::ProviderClient, embeddings::ImageEmbeddingModel, providers::cohere::Client,
};

#[tokio::main]
async fn main() -> Result<()> {
    let path = std::env::args_os()
        .nth(1)
        .context("pass the path to a PNG, JPEG, WebP, or GIF image")?;
    let image = std::fs::read(&path)
        .with_context(|| format!("failed to read image at {}", path.to_string_lossy()))?;

    let client = Client::from_env()?;
    let model = client.image_embedding_model();
    let embedding = model.embed_image(&image).await?;

    println!(
        "embedded {} bytes into {} dimensions",
        image.len(),
        embedding.vec.len()
    );

    Ok(())
}
