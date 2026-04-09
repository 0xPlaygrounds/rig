//! Migrated from `examples/perplexity_agent.rs`.

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::perplexity::{self, SONAR};

use crate::support::assert_nonempty_response;

#[tokio::test]
#[ignore = "requires PERPLEXITY_API_KEY"]
async fn completion_smoke() {
    let client = perplexity::Client::from_env();
    let agent = client
        .agent(SONAR)
        .preamble("Be precise and concise.")
        .temperature(0.5)
        .additional_params(serde_json::json!({
            "return_related_questions": true,
            "return_images": true
        }))
        .build();

    let response = agent
        .prompt("When and where and what type is the next solar eclipse?")
        .await
        .expect("prompt should succeed");

    assert_nonempty_response(&response);
}
