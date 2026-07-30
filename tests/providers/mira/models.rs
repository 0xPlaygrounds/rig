//! Migrated from `examples/agent_with_mira.rs`.

use rig::http_runtime::HttpRuntime;
use rig::providers::{mira, openai};

#[tokio::test]
#[ignore = "requires MIRA_API_KEY"]
async fn list_models_smoke() {
    let cfg = mira::functions::Config::from_env(openai::GPT_4O).expect("config should build");
    let rt = HttpRuntime::new();
    let models = mira::functions::list_models(&cfg, &rt)
        .await
        .expect("listing models should succeed");
    assert!(
        !models.data.is_empty(),
        "expected Mira to return at least one model"
    );
}
