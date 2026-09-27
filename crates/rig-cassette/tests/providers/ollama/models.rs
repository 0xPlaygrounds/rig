//! Ollama model listing smoke test.

use rig::providers::ollama::wire::OllamaConfig;
use rig_test_support::cassette_models::OllamaModels;

#[tokio::test]
#[ignore = "requires a local Ollama server"]
async fn list_models_smoke() {
    let ollama = OllamaModels::new(OllamaConfig::new(), rig::rig_reqwest::shared());
    let models = match ollama.list_models().await {
        Ok(models) => models,
        Err(error) => {
            panic!("listing Ollama models should succeed\nDisplay: {error}\nDebug: {error:#?}")
        }
    };

    assert!(
        !models.is_empty(),
        "expected Ollama to return at least one model\nModel list: {models:#?}"
    );
}
