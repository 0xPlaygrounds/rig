//! Ollama model listing smoke test.

use rig::model::ModelLister;
use rig::prelude::*;
use rig::providers::ollama::wire::Ollama;

#[tokio::test]
#[ignore = "requires a local Ollama server"]
async fn list_models_smoke() {
    let ollama = Ollama::new().bound().expect("transport should build");
    let models = match ollama.models().list_all().await {
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
