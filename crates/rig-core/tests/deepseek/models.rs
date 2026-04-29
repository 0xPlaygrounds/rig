//! DeepSeek model listing smoke test.
//!
//! Run with:
//! `cargo test -p rig-core --test deepseek deepseek::models::list_models_smoke -- --ignored --nocapture`

use rig_core::client::{ModelListingClient, ProviderClient};
use rig_core::providers::deepseek;

#[tokio::test]
#[ignore = "requires DEEPSEEK_API_KEY"]
async fn list_models_smoke() {
    let client = deepseek::Client::from_env().expect("client should build");
    let models = match client.list_models().await {
        Ok(models) => models,
        Err(error) => {
            panic!("listing DeepSeek models should succeed\nDisplay: {error}\nDebug: {error:#?}")
        }
    };

    assert!(
        !models.is_empty(),
        "expected DeepSeek to return at least one model\nModel list: {models:#?}"
    );

    assert!(
        models
            .iter()
            .any(|model| model.owned_by.as_deref() == Some("deepseek")),
        "expected at least one DeepSeek-owned model\nModel list: {models:#?}"
    );

    println!("DeepSeek returned {} models", models.len());
}
