//! Ollama model listing smoke test (`GET /api/tags`).
//!
//! Replays by default; set `RIG_PROVIDER_TEST_MODE=record` to record against a
//! local Ollama server.

use rig::client::ModelListingClient;

use super::super::support::with_ollama_cassette;

#[tokio::test]
async fn list_models_smoke() {
    with_ollama_cassette("models/list_models_smoke", |client| async move {
        let models = client
            .list_models()
            .await
            .expect("listing Ollama models should succeed");

        assert!(
            !models.is_empty(),
            "expected Ollama to return at least one model\nModel list: {models:#?}"
        );
    })
    .await;
}
