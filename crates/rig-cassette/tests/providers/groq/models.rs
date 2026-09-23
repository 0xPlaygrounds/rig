//! Groq model listing, recorded from the real API.
//!
//! Groq serves `GET /models` — the same path its credential check reads —
//! but for a long time rig declared no model-listing capability for it, so
//! listing Groq's catalogue was impossible (rig#2079). Recording the endpoint
//! proves the shared OpenAI-style envelope actually decodes against Groq's
//! wire, rather than assuming it because Groq is OpenAI-compatible.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use anyhow::Result;
use rig::error::ProviderError;
use rig::model::ModelLister;
use rig::providers::openai::wire::GROQ;

use super::support::{with_groq_cassette_bogus_key_result, with_groq_cassette_result};

#[tokio::test]
async fn list_models_smoke() -> Result<()> {
    with_groq_cassette_result("models/list_models_smoke", |client| async move {
        let models = client.models().list_all().await?;

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
/// the provider's reply, with its status and a route naming provider and path.
///
/// Pins that the Groq lister routes its failures through the same triage as
/// every other provider, not just its successes. The provider name and the
/// endpoint come from the dialect constant rather than from literals, so a
/// dialect that was renamed or repointed fails this cell instead of quietly
/// mislabelling its own errors. The recorded path carries the cassette
/// proxy's own prefix ahead of the endpoint, so the endpoint is matched as a
/// suffix.
#[tokio::test]
async fn list_models_rejected_key_reports_api_error_with_context() -> Result<()> {
    with_groq_cassette_bogus_key_result(
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
            let provider = format!("provider={}", GROQ.name);
            for expected in [
                provider.as_str(),
                GROQ.quirks.models_path,
                "status 401",
                "Invalid API Key",
            ] {
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
