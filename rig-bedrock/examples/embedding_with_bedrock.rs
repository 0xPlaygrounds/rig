use rig::Embed;
use rig::client::{EmbeddingsClient, ProviderClient};
use rig_bedrock::client::Client;
use rig_bedrock::embedding::AMAZON_TITAN_EMBED_TEXT_V2_0;
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

    let client = Client::from_env();
    let embeddings = client
        .embeddings_with_ndims(AMAZON_TITAN_EMBED_TEXT_V2_0, 256)
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
