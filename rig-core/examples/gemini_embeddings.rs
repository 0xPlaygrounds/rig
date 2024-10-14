use rig::providers::gemini::{self};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Initialize the Google Gemini client
    // Create OpenAI client
    let client = gemini::Client::from_env();

    let embeddings = client
        .embeddings(gemini::embedding::EMBEDDING_001)
        .simple_document("doc0", "Hello, world!")
        .simple_document("doc1", "Goodbye, world!")
        .build()
        .await
        .expect("Failed to embed documents");

    println!("{:?}", embeddings);

    Ok(())
}
