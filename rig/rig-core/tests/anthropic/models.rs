//! Anthropic model listing smoke test.

use rig::client::{ModelListingClient, ProviderClient};
use rig::providers::anthropic;

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn list_models_smoke() {
    let client = anthropic::Client::from_env();
    let models = client
        .list_models()
        .await
        .expect("listing Anthropic models should succeed");

    assert!(
        !models.is_empty(),
        "expected Anthropic to return at least one model"
    );
}
