//! Z.AI coding OpenAI-compatible completion smoke test.

use rig::providers::zai;
use rig::wire::Wire as _;

use crate::support::assert_nonempty_response;
use crate::zai::coding_client;

#[tokio::test]
#[ignore = "requires ZAI_API_KEY"]
async fn coding_openai_compatible_completion_smoke() {
    let response = rig::AgentBuilder::new(
        coding_client()
            .completion(zai::GLM_4_6)
            .on(rig::transport()),
    )
    .preamble("You are a concise coding assistant.")
    .build()
    .prompt("In one short sentence, explain what a unit test is.")
    .await
    .expect("Z.AI coding completion should succeed");

    assert_nonempty_response(&response.output);
}
