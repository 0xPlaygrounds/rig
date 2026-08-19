//! llama.cpp non-streaming tool round-trip.
//!
//! Replays by default; set `RIG_PROVIDER_TEST_MODE=record` to record against a
//! local OpenAI-compatible llama.cpp-family server (see `cassette_support`).

use rig::completion::{Chat, Message};
use rig::prelude::*;

use super::super::cassette_support::*;

use crate::support::{Adder, STREAMING_TOOLS_PREAMBLE, Subtract, assert_mentions_expected_number};
use rig::completion::Prompt;

#[tokio::test]
async fn tools_roundtrip() {
    with_llamacpp_cassette("tools/tools_roundtrip", |client| async move {
        let agent = client
            .agent(CASSETTE_MODEL)
            .preamble(STREAMING_TOOLS_PREAMBLE)
            .tool(Adder)
            .tool(Subtract)
            .default_max_turns(4)
            .build();

        let response = agent
            .chat("Calculate 2 - 5.", &mut Vec::<Message>::new())
            .await
            .expect("tool round-trip should succeed");

        assert_mentions_expected_number(&response, -3);
    })
    .await;
}

#[tokio::test]
async fn tools_smoke() {
    with_llamacpp_cassette("tools/tools_smoke", |client| async move {

        let agent = client
            .agent(CASSETTE_MODEL)
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

        assert_mentions_expected_number(&response, -3);
    })
    .await;
}
