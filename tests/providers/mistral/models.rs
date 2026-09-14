//! Mistral model listing, recorded from the real API.
//!
//! Was an `#[ignore]` live test, which meant Mistral's lister — one of the
//! providers routed through the shared `get_json` fetch — had no coverage that
//! runs in CI. rig#2079 recorded it, together with the failure path so the
//! shared status triage is pinned on this provider too.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use anyhow::Result;
use rig::client::ModelListingClient;

use super::support::{with_mistral_cassette_bogus_key_result, with_mistral_cassette_result};

#[tokio::test]
async fn list_models_smoke() -> Result<()> {
    with_mistral_cassette_result("models/list_models_smoke", |client| async move {
        let models = client.list_models().await?;

        anyhow::ensure!(
            !models.is_empty(),
            "expected Mistral to return at least one model",
        );
        anyhow::ensure!(
            models.data.iter().all(|model| !model.id.is_empty()),
            "every listing entry carries an id; got {:?}",
            models.data,
        );
        // The shared entry's optional fields must actually read Mistral's
        // wire: an all-`None` decode would look identical to a working one.
        anyhow::ensure!(
            models.data.iter().any(|model| model.owned_by.is_some()),
            "Mistral reports owned_by on its listing entries",
        );

        Ok::<_, anyhow::Error>(())
    })
    .await
}

/// rig#2079 — the shared fetch path classifies a rejected listing as
/// `ApiError` carrying provider, path, status and a body preview.
#[tokio::test]
async fn list_models_rejected_key_reports_api_error_with_context() -> Result<()> {
    use rig::model::ModelListingError;

    with_mistral_cassette_bogus_key_result(
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
            for expected in ["provider=Mistral", "path=/v1/models", "status=401"] {
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
