use rig_bedrock::client::BedrockRuntime;
use rig_bedrock::embedding::{AMAZON_TITAN_EMBED_TEXT_V2_0, Embeddings};
use rig_core::Model;
use rig_core::embeddings::EmbeddingsBuilder;
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

    let model = Model::new(
        Embeddings::new(AMAZON_TITAN_EMBED_TEXT_V2_0, Some(256)),
        BedrockRuntime::from_env()?,
    );
    let embeddings = EmbeddingsBuilder::new(model)
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
