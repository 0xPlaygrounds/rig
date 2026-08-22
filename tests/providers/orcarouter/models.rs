//! OrcaRouter model listing smoke test.

use rig::client::{ModelListingClient, ProviderClient};
use rig::providers::orcarouter;

#[tokio::test]
#[ignore = "requires ORCAROUTER_API_KEY"]
async fn list_models_smoke() {
    let client = orcarouter::Client::from_env().expect("client should build");
    let models = match client.list_models().await {
        Ok(models) => models,
        Err(error) => {
            panic!("listing OrcaRouter models should succeed\nDisplay: {error}\nDebug: {error:#?}")
        }
    };

    assert!(
        !models.is_empty(),
        "expected OrcaRouter to return at least one model\nModel list: {models:#?}"
    );
}
