use rig::providers::local;
use rig::Embed;

#[derive(Embed, Debug)]
struct Greetings {
    #[embed]
    message: String,
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Initialize the local client
    let client = local::Client::new();

    let embeddings = client
        .embeddings("mxbai-embed-large")
        .document(Greetings {
            message: "Hello, world!".to_string(),
        })?
        .document(Greetings {
            message: "Goodbye, world!".to_string(),
        })?
        .build()
        .await
        .expect("Failed to embed documents");

    println!("{:?}", embeddings);

    Ok(())
}
