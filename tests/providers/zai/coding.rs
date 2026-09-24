//! Z.AI coding OpenAI-compatible completion smoke test.

use rig::prelude::*;
use rig::providers::zai;

use crate::support::assert_nonempty_response;
use crate::zai::coding_client;

#[tokio::test]
#[ignore = "requires ZAI_API_KEY"]
async fn coding_openai_compatible_completion_smoke() {
    let response = coding_client()
        .endpoint(|provider_config| provider_config.completion(zai::GLM_4_6))
        .into_agent_builder()
        .preamble("You are a concise coding assistant.")
        .build()
        .prompt("In one short sentence, explain what a unit test is.")
        .await
        .expect("Z.AI coding completion should succeed");

    assert_nonempty_response(&response.output);
}
