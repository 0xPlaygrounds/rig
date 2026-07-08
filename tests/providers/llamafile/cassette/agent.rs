//! Llamafile agent completion smoke test.
//!
//! Replays by default; set `RIG_PROVIDER_TEST_MODE=record` to record against a
//! local OpenAI-compatible llama.cpp-family server (see `cassette_support`).

use rig::client::CompletionClient;
use rig::completion::Prompt;

use super::super::cassette_support::{CASSETTE_CHAT_MODEL, with_llamafile_cassette};
use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
async fn completion_smoke() {
    with_llamafile_cassette("agent/completion_smoke", |client| async move {
        let agent = client
            .agent(CASSETTE_CHAT_MODEL)
            .preamble(BASIC_PREAMBLE)
            .build();

        let response = agent
            .prompt(BASIC_PROMPT)
            .await
            .expect("completion should succeed");

        assert_nonempty_response(&response);
    })
    .await;
}
