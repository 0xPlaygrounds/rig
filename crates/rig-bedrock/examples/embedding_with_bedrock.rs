use rig_bedrock::client::Client;
use rig_bedrock::completion::AMAZON_TITAN_TEXT_EMBEDDINGS_V2;
use rig_core::client::{EmbeddingsClient, ProviderClient};
use tracing::info;

#[derive(rig_derive::Embed, Debug)]
struct Greetings {
    #[embed]
    message: String,
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    let client = Client::from_env()?;
    let embeddings = client
        .embeddings_with_ndims(AMAZON_TITAN_TEXT_EMBEDDINGS_V2, 256)
        .document(Greetings {
            message: "aa".to_string(),
        })?
        .document(Greetings {
            message: "bb".to_string(),
        })?
        .build()
        .await?;

    info!("{:?}", embeddings);

    Ok(())
}
