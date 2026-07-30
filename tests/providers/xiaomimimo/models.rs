//! Xiaomi MiMo model listing smoke test.

use rig::http_runtime::HttpRuntime;
use rig::providers::xiaomimimo::{
    self, MIMO_V2_5, MIMO_V2_5_PRO, MIMO_V2_FLASH, MIMO_V2_OMNI, MIMO_V2_PRO,
};

#[tokio::test]
#[ignore = "requires XIAOMI_MIMO_API_KEY"]
async fn list_models_smoke() {
    let cfg = xiaomimimo::functions::Config::from_env(MIMO_V2_5_PRO).expect("config should build");
    let rt = HttpRuntime::new();
    let models = match xiaomimimo::functions::list_models(&cfg, &rt).await {
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
