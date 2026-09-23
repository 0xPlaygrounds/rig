//! DeepSeek model listing smoke test.
//!
//! Run with:
//! `cargo test -p rig --test deepseek list_models_smoke -- --ignored --nocapture`

use rig::error::ProviderError;
use rig::model::ModelLister;

use super::support::with_deepseek_cassette;

#[tokio::test]
async fn list_models_smoke() {
    with_deepseek_cassette("models/list_models_smoke", |client| async move {
        let models = match client.models().list_all().await {
            Ok(models) => models,
            Err(error) => {
                panic!(
                    "listing DeepSeek models should succeed\nDisplay: {error}\nDebug: {error:#?}"
                )
            }
        };

        assert!(
            !models.is_empty(),
            "expected DeepSeek to return at least one model\nModel list: {models:#?}"
        );

        assert!(
            models
                .iter()
                .any(|model| model.owned_by.as_deref() == Some("deepseek")),
            "expected at least one DeepSeek-owned model\nModel list: {models:#?}"
        );

        println!("DeepSeek returned {} models", models.len());
    })
    .await;
}

/// rig#2079 — the shared fetch path classifies a rejected listing as
/// the provider's reply, with its status and a route naming provider and path.
#[tokio::test]
async fn list_models_rejected_key_reports_api_error_with_context() -> anyhow::Result<()> {
    super::support::with_deepseek_cassette_bogus_key_result(
        "models/list_models_rejected_key_reports_api_error_with_context",
        |client| async move {
            let error = client
                .models()
                .list_all()
                .await
                .expect_err("a bogus key must not list models");

            let ProviderError::ProviderResponse(response) = &error else {
                anyhow::bail!(
                    "a rejected listing must keep the provider's reply\nDisplay: {error}\n\
                     Debug: {error:#?}"
                );
            };
            let status_code = response.status.map(|status| status.as_u16());
            let message = error.to_string();

            anyhow::ensure!(status_code == Some(401), "unexpected status: {error:#?}");
            // `provider=` is the wire's descriptor name — the same `deepseek`
            // every normalized response reports, not a display label. The
            // deleted client layer decorated this message with its own
            // capitalised `DeepSeek` instead, so one provider's identity had
            // two spellings; there is now one. `path=` is unchanged: DeepSeek's
            // base URL carries no version segment, so the listing really does
            // send `/models`.
            for expected in ["provider=deepseek", "path=/models", "status 401"] {
                anyhow::ensure!(
                    message.contains(expected),
                    "the error must carry {expected}; got {message}"
                );
            }
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}
