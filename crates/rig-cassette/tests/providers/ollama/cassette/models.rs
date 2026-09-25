//! Ollama model listing smoke test (`GET /api/tags`).
//!
//! Replays by default; set `RIG_PROVIDER_TEST_MODE=record` to record against a
//! local Ollama server.

use super::super::support::with_ollama_cassette;
use rig::wire::Wire as _;

#[tokio::test]
async fn list_models_smoke() {
    with_ollama_cassette("models/list_models_smoke", |client| async move {
        let models = client
            .models()
            .on(rig::transport())
            .call(())
            .await
            .expect("listing Ollama models should succeed");

        assert!(
            !models.is_empty(),
            "expected Ollama to return at least one model\nModel list: {models:#?}"
        );
    })
    .await;
}
