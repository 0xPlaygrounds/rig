//! Groq tools smoke test.

use rig::prelude::*;
use rig::providers::openai::wire::{GROQ, OpenAI};

use crate::support::{Adder, Subtract, assert_mentions_expected_number};

use super::TOOLS_MODEL;

#[tokio::test]
#[ignore = "requires GROQ_API_KEY"]
async fn tools_smoke() {
    let groq = OpenAI::from_env_with(&GROQ)
        .expect("GROQ_API_KEY should be set")
        .bound()
        .expect("transport should build");
    let agent = groq
        .agent(TOOLS_MODEL)
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
