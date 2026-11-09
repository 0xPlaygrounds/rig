//! Cassette-backed OpenRouter model listing smoke test.

/// The entry's `context_length` must survive decoding, and
/// `max_output_tokens` comes from `top_provider.max_completion_tokens`.
/// OpenRouter sends snake_case keys, so a camelCase DTO would report `None`.
#[tokio::test]
async fn list_models_preserves_context_and_output_limits() -> anyhow::Result<()> {
    super::super::support::with_openrouter_cassette_result(
        "models/list_models_smoke",
        |client| async move {
            let models = client.list_models().await?;

            anyhow::ensure!(
                models
                    .data
                    .iter()
                    .any(|model| model.context_length.is_some()),
                "OpenRouter reports context_length on its listing entries",
            );
            anyhow::ensure!(
                models
                    .data
                    .iter()
                    .any(|model| model.max_output_tokens.is_some()),
                "OpenRouter reports top_provider.max_completion_tokens",
            );
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}
