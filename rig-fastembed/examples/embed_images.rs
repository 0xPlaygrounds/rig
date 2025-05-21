use rig::embeddings::embedding::ImageEmbeddingModel as _;
use rig_fastembed::FastembedImageModel;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let fastembed_client = rig_fastembed::Client::new();

    let embedding_model =
        fastembed_client.image_embedding_model(&FastembedImageModel::NomicEmbedVisionV15);

    let bytes = std::fs::read("image.png").unwrap();

    let res = embedding_model.embed_image(&bytes).await.unwrap();

    println!("{}", res.vec.len());

    Ok(())
}
