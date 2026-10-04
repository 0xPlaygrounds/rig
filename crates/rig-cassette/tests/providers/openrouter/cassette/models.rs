//! Cassette-backed OpenRouter model listing smoke test.

/// rig#2079 — the entry's `context_length` must survive decoding.
///
/// A stray `rename_all = "camelCase"` on the DTO made serde look for
/// `contextLength`, which OpenRouter never sends, so every model reported
/// `None` while the response carried a real window. `max_output_tokens` comes
/// from `top_provider.max_completion_tokens`, which the DTO previously did not
/// read at all.
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
