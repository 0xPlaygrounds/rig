//! DeepSeek model listing smoke test.
//!
//! Run with:
//! `cargo test -p rig --test deepseek list_models_smoke -- --ignored --nocapture`

use rig::client::ModelListingClient;

use super::support::with_deepseek_cassette;

#[tokio::test]
async fn list_models_smoke() {
    with_deepseek_cassette("models/list_models_smoke", |client| async move {
        let models = match client.list_models().await {
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
/// `ApiError` carrying provider, path, status and a body preview.
#[tokio::test]
async fn list_models_rejected_key_reports_api_error_with_context() -> anyhow::Result<()> {
    use rig::model::ModelListingError;

    super::support::with_deepseek_cassette_bogus_key_result(
        "models/list_models_rejected_key_reports_api_error_with_context",
        |client| async move {
            let error = client
                .list_models()
                .await
                .expect_err("a bogus key must not list models");

            let ModelListingError::ApiError {
                status_code,
                message,
            } = &error
            else {
                anyhow::bail!(
                    "a rejected listing must classify as ApiError\nDisplay: {error}\n\
                     Debug: {error:#?}"
                );
            };

            anyhow::ensure!(*status_code == 401, "unexpected status: {error:#?}");
            for expected in ["provider=DeepSeek", "path=/models", "status=401"] {
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
