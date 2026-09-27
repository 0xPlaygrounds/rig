//! Groq tools smoke test.

use rig::providers::openai::wire::GROQ;
use rig_test_support::cassette_models::OpenAiModels;

use crate::support::{Adder, Subtract, assert_mentions_expected_number};

use super::TOOLS_MODEL;

#[tokio::test]
#[ignore = "requires GROQ_API_KEY"]
async fn tools_smoke() {
    let groq = OpenAiModels::from_env_for(&GROQ).expect("GROQ_API_KEY should be set");
    let agent = rig::AgentBuilder::new(groq
        .completion(TOOLS_MODEL))
        .preamble(
            "You are a calculator. For arithmetic requests, call the appropriate tool exactly once. \
             After you receive the tool result, do not call any more tools and reply with the final numeric answer only.",
        )
        .tool(Adder)
        .tool(Subtract)
        .build();

    let response = agent
        .prompt("Calculate 2 - 5. Call `subtract` exactly once, then answer with just the result.")
        .max_turns(3)
        .await
        .expect("tool prompt should succeed");

    assert_mentions_expected_number(&response.output, -3);
}
