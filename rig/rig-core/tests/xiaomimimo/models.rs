//! Xiaomi MiMo model listing smoke test.

use rig::client::{ModelListingClient, ProviderClient};
use rig::providers::xiaomimimo;

#[tokio::test]
#[ignore = "requires XIAOMI_MIMO_API_KEY"]
async fn list_models_smoke() {
    let client = xiaomimimo::Client::from_env().expect("client should build");
    let models = match client.list_models().await {
        Ok(models) => models,
        Err(error) => {
            panic!(
                "listing Xiaomi MiMo models should succeed\nDisplay: {error}\nDebug: {error:#?}"
            )
        }
    };

    assert!(
        !models.is_empty(),
        "expected Xiaomi MiMo to return at least one model\nModel list: {models:#?}"
    );

    assert!(
        models
            .iter()
            .any(|model| model.owned_by.as_deref() == Some("xiaomi")),
        "expected at least one Xiaomi-owned model\nModel list: {models:#?}"
    );
}
