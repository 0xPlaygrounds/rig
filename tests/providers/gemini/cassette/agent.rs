//! Gemini agent completion smoke test.

use rig::completion::Prompt;
use rig::prelude::*;
use rig::providers::gemini;

use super::super::support::with_gemini_cassette;
use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
async fn completion_smoke() {
    with_gemini_cassette("agent/completion_smoke", |client| async move {
        let agent = client
            .agent(gemini::completion::GEMINI_2_5_FLASH)
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
