//! llama.cpp model listing smoke test.

use rig_core::client::ModelListingClient;

use super::support;

#[tokio::test]
#[ignore = "requires a local llama.cpp OpenAI-compatible server"]
async fn list_models_smoke() {
    let client = support::client();
    let models = match client.list_models().await {
        Ok(models) => models,
        Err(error) => {
            panic!("listing llama.cpp models should succeed\nDisplay: {error}\nDebug: {error:#?}")
        }
    };

    assert!(
        !models.is_empty(),
        "expected llama.cpp to return at least one model\nModel list: {models:#?}"
    );
}
