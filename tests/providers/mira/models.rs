//! Migrated from `examples/agent_with_mira.rs`.

use rig::providers::openai::wire::{MIRA, OpenAI};
use rig_test_support::endpoint::Endpoint;

#[tokio::test]
#[ignore = "requires MIRA_API_KEY"]
async fn list_models_smoke() {
    let provider = Endpoint::new(
        OpenAI::from_env_with(&MIRA).expect("config should build from env"),
        rig::rig_reqwest::shared(),
    );
    let models = provider
        .models()
        .call((), None)
        .await
        .expect("listing models should succeed");
    assert!(
        !models.is_empty(),
        "expected Mira to return at least one model"
    );
}
