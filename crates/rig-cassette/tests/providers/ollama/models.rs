//! Ollama model listing smoke test.

use rig::providers::ollama::wire::Ollama;
use rig::wire::Wire as _;

#[tokio::test]
#[ignore = "requires a local Ollama server"]
async fn list_models_smoke() {
    let ollama = Ollama::new();
    let models = match ollama.models().on(rig::transport()).call(()).await {
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
