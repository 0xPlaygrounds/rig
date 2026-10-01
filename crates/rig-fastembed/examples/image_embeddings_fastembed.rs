use anyhow::{Context, ensure};
use rig_fastembed::{FastembedImage, FastembedImageModel};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let paths: Vec<_> = std::env::args_os().skip(1).collect();
    ensure!(
        !paths.is_empty(),
        "Usage: cargo run -p rig-fastembed --example image_embeddings_fastembed -- <image>..."
    );
    let images = paths
        .iter()
        .map(|path| std::fs::read(path).with_context(|| format!("Could not read {path:?}")))
        .collect::<anyhow::Result<Vec<_>>>()?;

    let runtime = FastembedImage::load(&FastembedImageModel::ClipVitB32)?;
    let model = runtime.embedding(&FastembedImageModel::ClipVitB32, None);
    let response = model.call(images).await?;

    for (path, embedding) in paths.iter().zip(response.embeddings) {
        println!(
            "{path:?}: {} dimensions ({})",
            embedding.vec.len(),
            embedding.document
        );
    }
    Ok(())
}
