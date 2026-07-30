//! Ollama model listing smoke test (`GET /api/tags`).
//!
//! Replays by default; set `RIG_PROVIDER_TEST_MODE=record` to record against a
//! local Ollama server.

use rig::http_runtime::HttpRuntime;
use rig::providers::ollama;

use super::super::support::with_ollama_cassette;

#[tokio::test]
async fn list_models_smoke() {
    with_ollama_cassette("models/list_models_smoke", |env| async move {
        let cfg = env.config("qwen3:4b");
        let rt = HttpRuntime::new();
        let models = ollama::functions::list_models(&cfg, &rt)
            .await
            .expect("listing Ollama models should succeed");

        assert!(
            !models.is_empty(),
            "expected Ollama to return at least one model\nModel list: {models:#?}"
        );
    })
    .await;
}
