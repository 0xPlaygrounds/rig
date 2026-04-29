//! OpenAI model listing smoke test.

use rig_core::client::{ModelListingClient, ProviderClient};
use rig_core::providers::openai;

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn list_models_smoke() {
    let client = openai::Client::from_env().expect("client should build");
    let models = match client.list_models().await {
        Ok(models) => models,
        Err(error) => {
            panic!("listing OpenAI models should succeed\nDisplay: {error}\nDebug: {error:#?}")
        }
    };

    assert!(
        !models.is_empty(),
        "expected OpenAI to return at least one model\nModel list: {models:#?}"
    );
}
