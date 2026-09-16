//! Migrated from `examples/perplexity_agent.rs`.

use rig::prelude::*;
use rig::providers::openai::wire::{OpenAI, PERPLEXITY};
use rig::providers::perplexity::SONAR;

use crate::support::assert_nonempty_response;

#[tokio::test]
#[ignore = "requires PERPLEXITY_API_KEY"]
async fn completion_smoke() {
    let perplexity = OpenAI::from_env_with(&PERPLEXITY)
        .expect("config should build from env")
        .bound()
        .expect("transport should build");
    let agent = perplexity
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
        .expect("prompt should succeed")
        .output;

    assert_nonempty_response(&response);
}
