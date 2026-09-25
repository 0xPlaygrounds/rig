//! Migrated from `examples/agent_with_mira.rs`.

use rig::providers::openai::wire::{MIRA, OpenAI};
use rig::wire::Wire as _;

#[tokio::test]
#[ignore = "requires MIRA_API_KEY"]
async fn list_models_smoke() {
    let provider = OpenAI::from_env_with(&MIRA).expect("config should build from env");
    let models = provider
        .models()
        .on(rig::transport())
        .call(())
        .await
        .expect("listing models should succeed");
    assert!(
        !models.is_empty(),
        "expected Mira to return at least one model"
    );
}
