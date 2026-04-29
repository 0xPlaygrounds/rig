//! Xiaomi MiMo model listing smoke test.

use rig_core::client::{ModelListingClient, ProviderClient};
use rig_core::providers::xiaomimimo::{
    self, MIMO_V2_5, MIMO_V2_5_PRO, MIMO_V2_FLASH, MIMO_V2_OMNI, MIMO_V2_PRO,
};

#[tokio::test]
#[ignore = "requires XIAOMI_MIMO_API_KEY"]
async fn list_models_smoke() {
    let client = xiaomimimo::Client::from_env().expect("client should build");
    let models = match client.list_models().await {
        Ok(models) => models,
        Err(error) => {
            panic!("listing Xiaomi MiMo models should succeed\nDisplay: {error}\nDebug: {error:#?}")
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

    let model_ids: Vec<&str> = models.iter().map(|m| m.id.as_str()).collect();

    for expected_id in [
        MIMO_V2_FLASH,
        MIMO_V2_OMNI,
        MIMO_V2_PRO,
        MIMO_V2_5,
        MIMO_V2_5_PRO,
    ] {
        assert!(
            model_ids.contains(&expected_id),
            "expected model {expected_id:?} in response\nReturned model IDs: {model_ids:#?}"
        );
    }
}
