//! Migrated from `examples/agent_with_mira.rs`.

use rig::providers::mira;

#[tokio::test]
#[ignore = "requires MIRA_API_KEY"]
async fn list_models_smoke() {
    let provider = mira::from_env().expect("config should build from env");
    let models = provider
        .list_models()
        .await
        .expect("listing models should succeed");
    assert!(
        !models.is_empty(),
        "expected Mira to return at least one model"
    );
}
