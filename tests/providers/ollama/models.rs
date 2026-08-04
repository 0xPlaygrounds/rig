//! Ollama model listing smoke test.

use rig::http_runtime::HttpRuntime;
use rig::providers::ollama;

#[tokio::test]
#[ignore = "requires a local Ollama server"]
async fn list_models_smoke() {
    let cfg = ollama::functions::Config::new("qwen3:4b");
    let rt = HttpRuntime::new();
    let models = match ollama::functions::list_models(&cfg, &rt).await {
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
