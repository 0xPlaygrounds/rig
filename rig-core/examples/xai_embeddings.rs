use rig::providers::xai;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Initialize the xAI client
    let client = xai::Client::from_env();

    let embeddings = client
        .embeddings(xai::embedding::EMBEDDING_V1)
        .simple_document("doc0", "Hello, world!")
        .simple_document("doc1", "Goodbye, world!")
        .build()
        .await
        .expect("Failed to embed documents");

    println!("{:?}", embeddings);

    Ok(())
}
