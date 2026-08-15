//! Groq model listing, recorded from the real API.
//!
//! Groq serves `GET /models` — the same path its client already uses for
//! `VerifyClient::verify` — but declared no `model_listing` capability, so
//! `list_models()` was unavailable (rig#2079). Recording the endpoint proves
//! the shared OpenAI-style envelope actually decodes against Groq's wire,
//! rather than assuming it because Groq is OpenAI-compatible.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use anyhow::Result;
use rig::client::ModelListingClient;

use super::support::{with_groq_cassette_bogus_key_result, with_groq_cassette_result};

#[tokio::test]
async fn list_models_smoke() -> Result<()> {
    with_groq_cassette_result("models/list_models_smoke", |client| async move {
        let models = client.list_models().await?;

        anyhow::ensure!(
            !models.data.is_empty(),
            "Groq should return at least one model",
        );
        anyhow::ensure!(
            models.data.iter().all(|model| !model.id.is_empty()),
            "every listing entry carries an id; got {:?}",
            models.data,
        );
        // Pin every optional field Groq actually reports. A DTO that decoded
        // into all-`None` optionals would otherwise look identical to a
        // working one — which is exactly how a provider-reported output
        // ceiling got dropped on the floor before (rig#2322). `created` is
        // excluded: the recorder scrubs timestamps to 0.
        for (field, present) in [
            ("name", models.data.iter().any(|m| m.name.is_some())),
            ("owned_by", models.data.iter().any(|m| m.owned_by.is_some())),
            (
                "context_length",
                models.data.iter().any(|m| m.context_length.is_some()),
            ),
            (
                "max_output_tokens",
                models.data.iter().any(|m| m.max_output_tokens.is_some()),
            ),
        ] {
            anyhow::ensure!(present, "Groq reports {field} on its listing entries");
        }

        Ok::<_, anyhow::Error>(())
    })
    .await
}

/// rig#2079 — the shared fetch path classifies a rejected listing as
/// `ApiError` carrying provider, path, status and a body preview.
///
/// Pins that the newly added Groq lister routes its failures through the same
/// triage as every other provider, not just its successes.
#[tokio::test]
async fn list_models_rejected_key_reports_api_error_with_context() -> Result<()> {
    use rig::model::ModelListingError;

    with_groq_cassette_bogus_key_result(
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
            for expected in ["provider=Groq", "path=/models", "status=401"] {
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
