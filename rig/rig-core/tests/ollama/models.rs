//! Ollama model listing smoke test.

use rig::client::{ModelListingClient, Nothing};
use rig::providers::ollama;

#[tokio::test]
#[ignore = "requires a local Ollama server"]
async fn list_models_smoke() {
    let client = ollama::Client::new(Nothing).expect("client should build");
    let models = match client.list_models().await {
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
