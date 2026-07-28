use rig::{embeddings::EmbeddingModel, prelude::*, providers::azure::Client};

/// Azure's embedding client must return the provider's token usage, not zeroes.
///
/// `EmbeddingModel::embed_texts_with_usage` has a default body that discards
/// usage and returns `Usage::default()`. Azure did not override it, so callers
/// reading `EmbeddingResponse::usage` saw zero tokens on every request — while
/// `embed_texts` had already parsed `response.usage` and passed it to
/// `tracing::info!` before dropping it.
///
/// A live test rather than a cassette one, matching this provider's existing
/// convention (`structured_output.rs`, `transcription.rs`): there is no
/// `tests/cassettes/azure/` fixture set or helper to extend.
#[tokio::test]
#[ignore = "requires AZURE_OPENAI_API_KEY and related Azure env vars"]
async fn test_azure_embeddings_report_token_usage() -> anyhow::Result<()> {
    let _ = tracing_subscriber::fmt::try_init();

    let client = Client::from_env()?;
    let model = client.embedding_model("text-embedding-3-small");

    let response = model
        .embed_texts_with_usage(vec![
            "Embeddings turn text into numeric vectors for similarity search.".to_string(),
        ])
        .await?;

    anyhow::ensure!(
        response.embeddings.len() == 1,
        "expected one embedding, got {}",
        response.embeddings.len()
    );
    anyhow::ensure!(
        !response.embeddings[0].vec.is_empty(),
        "embedding vector must not be empty"
    );
    anyhow::ensure!(
        response.usage.input_tokens > 0,
        "Azure reports prompt_tokens for every embedding request, so input_tokens \
         must be non-zero; got {} — this is the regression that made every \
         downstream cost figure read zero",
        response.usage.input_tokens
    );

    tracing::info!("Azure embedding usage: {:?}", response.usage);
    Ok(())
}
