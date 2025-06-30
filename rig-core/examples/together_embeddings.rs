use rig::Embed;
use rig::prelude::*;
use rig::providers::together;

#[derive(Embed, Debug)]
struct Greetings {
    #[embed]
    message: String,
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Initialize the together client
    let client = together::Client::from_env();
    let embeddings = client
        .embeddings(together::embedding::M2_BERT_80M_8K_RETRIEVAL)
        .document(Greetings {
            message: "Hello, world!".to_string(),
        })?
        .document(Greetings {
            message: "Goodbye, world!".to_string(),
        })?
        .build()
        .await
        .expect("Failed to embed documents");

    println!("{embeddings:?}");

    Ok(())
}
