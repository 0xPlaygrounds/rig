//! Gemini agent completion smoke test.

use rig::providers::gemini;
use rig::wire::Wire as _;

use super::super::support::with_gemini_cassette;
use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
async fn completion_smoke() {
    with_gemini_cassette("agent/completion_smoke", |client| async move {
        let agent = rig::AgentBuilder::new(
            client
                .completion(gemini::completion::GEMINI_2_5_FLASH)
                .on(rig::transport()),
        )
        .preamble(BASIC_PREAMBLE)
        .build();

        let response = agent
            .prompt(BASIC_PROMPT)
            .await
            .expect("completion should succeed");

        assert_nonempty_response(&response.output);
    })
    .await;
}
