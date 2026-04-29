//! OpenRouter model listing smoke test.
//!
//! Run with:
//! `cargo test -p rig-core --test openrouter openrouter::models::list_models_smoke -- --ignored --nocapture`

use rig_core::client::{ModelListingClient, ProviderClient};
use rig_core::providers::openrouter;

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn list_models_smoke() {
    let client = openrouter::Client::from_env().expect("client should build");
    let models = match client.list_models().await {
        Ok(models) => models,
        Err(error) => {
            panic!("listing OpenRouter models should succeed\nDisplay: {error}\nDebug: {error:#?}")
        }
    };

    assert!(
        !models.is_empty(),
        "expected OpenRouter to return at least one model\nModel list: {models:#?}"
    );

    println!("OpenRouter returned {} models", models.len());
}
