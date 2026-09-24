//! Z.AI Anthropic-compatible completion smoke test.

use rig::prelude::*;
use rig::providers::zai;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};
use crate::zai::anthropic_client;

#[tokio::test]
#[ignore = "requires ZAI_API_KEY"]
async fn anthropic_compatible_completion_smoke() {
    let response = anthropic_client()
        .endpoint(|provider_config| provider_config.completion(zai::GLM_4_6))
        .into_agent_builder()
        .preamble(BASIC_PREAMBLE)
        .build()
        .prompt(BASIC_PROMPT)
        .await
        .expect("Z.AI Anthropic-compatible completion should succeed");

    assert_nonempty_response(&response.output);
}
