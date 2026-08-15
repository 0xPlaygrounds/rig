//! Cassette-backed OpenRouter model listing smoke test.

use rig::client::ModelListingClient;

use super::super::support::with_openrouter_cassette;

#[tokio::test]
async fn list_models_smoke() {
    with_openrouter_cassette("models/list_models_smoke", |client| async move {
        let models = match client.list_models().await {
            Ok(models) => models,
            Err(error) => {
                panic!(
                    "listing OpenRouter models should succeed\nDisplay: {error}\nDebug: {error:#?}"
                )
            }
        };

        assert!(
            !models.is_empty(),
            "expected OpenRouter to return at least one model\nModel list: {models:#?}"
        );
    })
    .await;
}

/// rig#2079 — OpenRouter's listing is **public**: a rejected key still lists.
///
/// Recorded with a deliberately invalid key. This is not an error cell that
/// stopped working — it is the provider's real behavior, pinned so nobody
/// later "fixes" the shared fetch on the assumption that every listing needs
/// auth.
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

#[tokio::test]
async fn list_models_is_public_and_ignores_a_rejected_key() -> anyhow::Result<()> {
    super::super::support::with_openrouter_cassette_bogus_key_result(
        "models/list_models_is_public_and_ignores_a_rejected_key",
        |client| async move {
            let models = client
                .list_models()
                .await
                .expect("OpenRouter lists models without a valid key");

            anyhow::ensure!(!models.is_empty(), "expected a non-empty public listing");
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}
