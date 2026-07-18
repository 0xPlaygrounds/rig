//! Llamafile non-streaming tool round-trip.
//!
//! Replays by default; set `RIG_PROVIDER_TEST_MODE=record` to record against a
//! local OpenAI-compatible llama.cpp-family server (see `cassette_support`).

use rig::client::AgentClientExt;
use rig::completion::{Chat, Message};

use super::super::cassette_support::{CASSETTE_CHAT_MODEL, with_llamafile_cassette};
use crate::support::{Adder, STREAMING_TOOLS_PREAMBLE, Subtract, assert_mentions_expected_number};

#[tokio::test]
async fn tools_roundtrip() {
    with_llamafile_cassette("tools/tools_roundtrip", |client| async move {
        let agent = client
            .agent(CASSETTE_CHAT_MODEL)
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
