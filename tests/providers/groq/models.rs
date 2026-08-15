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

use super::support::with_groq_cassette_result;

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
        // `owned_by` is the optional field of the shared entry DTO that Groq
        // does populate. Asserting it pins the DTO as *reading* Groq's wire
        // rather than merely tolerating it — a listing that decoded into all
        // `None` optionals would otherwise look identical to a working one.
        anyhow::ensure!(
            models.data.iter().any(|model| model.owned_by.is_some()),
            "Groq populates owned_by on its listing entries",
        );

        Ok::<_, anyhow::Error>(())
    })
    .await
}
