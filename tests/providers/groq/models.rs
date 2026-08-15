//! Groq model listing, recorded from the real API.
//!
//! Groq exposes `GET /models` — the same path its client already uses for
//! `VerifyClient::verify` — but declared no `model_listing` capability, so
//! `list_models()` was unavailable (rig#2079). This records the endpoint to
//! prove the shared OpenAI-style envelope actually decodes against Groq's
//! wire, rather than assuming it because Groq is OpenAI-compatible.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use rig::client::ModelListingClient;

use super::support::with_groq_cassette_result;

#[tokio::test]
async fn list_models_smoke() -> Result<(), Box<dyn std::error::Error>> {
    with_groq_cassette_result("models/list_models_smoke", |client| async move {
        let models = client.list_models().await?;

        assert!(
            !models.data.is_empty(),
            "Groq should return at least one model",
        );
        for model in &models.data {
            assert!(!model.id.is_empty(), "every model carries an id: {model:?}");
        }
        // `owned_by` is the optional field of the shared entry that Groq does
        // populate; asserting it pins that the shared DTO reads Groq's wire
        // rather than merely tolerating it.
        assert!(
            models.data.iter().any(|model| model.owned_by.is_some()),
            "Groq populates owned_by on its listing entries",
        );

        Ok::<_, Box<dyn std::error::Error>>(())
    })
    .await
}
