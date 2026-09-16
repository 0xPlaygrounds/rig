//! Migrated from `examples/agent_with_mira.rs`.

use rig::model::ModelLister;
use rig::prelude::*;
use rig::providers::openai::wire::{MIRA, OpenAI};

#[tokio::test]
#[ignore = "requires MIRA_API_KEY"]
async fn list_models_smoke() {
    let provider = OpenAI::from_env_with(&MIRA)
        .expect("config should build from env")
        .bound()
        .expect("transport should build");
    let models = provider
        .models()
        .list_all()
        .await
        .expect("listing models should succeed");
    assert!(
        !models.is_empty(),
        "expected Mira to return at least one model"
    );
}
