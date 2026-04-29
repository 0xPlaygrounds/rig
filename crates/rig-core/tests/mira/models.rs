//! Migrated from `examples/agent_with_mira.rs`.

use rig_core::client::ProviderClient;
use rig_core::providers::mira;

#[tokio::test]
#[ignore = "requires MIRA_API_KEY"]
async fn list_models_smoke() {
    let client = mira::Client::from_env().expect("client should build");
    let models = client
        .list_models()
        .await
        .expect("listing models should succeed");
    assert!(
        !models.is_empty(),
        "expected Mira to return at least one model"
    );
}
