use super::*;
use crate::test_utils::RecordingHttpClient;

/// The reply shape of `crates/rig-cassette/fixtures/cassettes/ollama/models/list_models_smoke.yaml`
/// (`GET /api/tags`), with two of its entries.
const MODELS_BODY: &str = r#"{"models":[{"name":"all-minilm:latest","model":"all-minilm:latest","modified_at":"2026-06-19T17:15:40.188240254-07:00","size":45960996},{"name":"qwen3:4b","model":"qwen3:4b","modified_at":"2026-06-19T16:26:52.429441648-07:00","size":2497293931}]}"#;

#[tokio::test]
async fn the_model_listing_reads_every_installed_model() {
    let models = crate::driver::Model::new(
        OllamaConfig::new().models(),
        RecordingHttpClient::new(MODELS_BODY),
    )
    .list()
    .await
    .expect("the recorded reply decodes");

    assert_eq!(
        models
            .data
            .iter()
            .map(|model| model.id.as_str())
            .collect::<Vec<_>>(),
        vec!["all-minilm:latest", "qwen3:4b"]
    );
}
