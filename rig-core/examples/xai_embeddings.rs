use rig::providers::xai;
use rig::Embed;

#[derive(Embed, Debug)]
struct Greetings {
    id: String,
    #[embed]
    message: String,
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Initialize the xAI client
    let client = xai::Client::from_env();

    let embeddings = client
        .embeddings(xai::embedding::EMBEDDING_V1)
        .document(Greetings {
            id: "doc0".to_string(),
            message: "Hello, world!".to_string(),
        })?
        .document(Greetings {
            id: "doc1".to_string(),
            message: "Goodbye, world!".to_string(),
        })?
        .build()
        .await
        .expect("Failed to embed documents");

    println!("{:?}", embeddings);

    Ok(())
}
