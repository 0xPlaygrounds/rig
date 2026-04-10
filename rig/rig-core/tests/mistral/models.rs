//! Mistral model listing smoke test.

use rig::client::{ModelListingClient, ProviderClient};
use rig::providers::mistral;

#[tokio::test]
#[ignore = "requires MISTRAL_API_KEY"]
async fn list_models_smoke() {
    let client = mistral::Client::from_env();
    let models = match client.list_models().await {
        Ok(models) => models,
        Err(error) => {
            panic!("listing Mistral models should succeed\nDisplay: {error}\nDebug: {error:#?}")
        }
    };

    assert!(
        !models.is_empty(),
        "expected Mistral to return at least one model\nModel list: {models:#?}"
    );
}
