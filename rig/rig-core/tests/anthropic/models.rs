//! Anthropic model listing smoke test.

use rig::client::{ModelListingClient, ProviderClient};
use rig::providers::anthropic;

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn list_models_smoke() {
    let client = anthropic::Client::from_env();
    let models = match client.list_models().await {
        Ok(models) => models,
        Err(error) => {
            panic!("listing Anthropic models should succeed\nDisplay: {error}\nDebug: {error:#?}")
        }
    };

    assert!(
        !models.is_empty(),
        "expected Anthropic to return at least one model\nModel list: {models:#?}"
    );
}
