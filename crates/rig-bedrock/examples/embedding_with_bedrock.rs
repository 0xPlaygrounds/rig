//! Embedding documents with Bedrock.
//!
//! `EmbeddingsBuilder` and the `EmbeddingsClient` trait are gone: the
//! embedding face is a plain `functions::EmbeddingConfig`, an AWS client built
//! from it, and `rig_core::embeddings::embed_documents`, which chunks the
//! `#[embed]` texts to the provider's `max_embedding_documents` and hands each
//! chunk to `functions::embed`.
use rig_bedrock::embedding::AMAZON_TITAN_EMBED_TEXT_V2_0;
use rig_bedrock::functions;
use rig_core::embeddings::EmbeddingJob;
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

    // Bedrock authenticates through the AWS SDK's default credential chain,
    // so the provider is expressed as plain configuration.
    let cfg = functions::EmbeddingConfig::new(AMAZON_TITAN_EMBED_TEXT_V2_0).with_ndims(256);
    let client = functions::client_from_config(&cfg.client_config()).await;

    let embeddings = EmbeddingJob::new()
        .documents(vec![
            Greetings {
                message: "aa".to_string(),
            },
            Greetings {
                message: "bb".to_string(),
            },
        ])
        .for_provider(&functions::DESCRIPTOR)
        .run(|texts| functions::embed(&client, &cfg.model, cfg.ndims, texts))
        .await?;

    info!("{:?}", embeddings);

    Ok(())
}
