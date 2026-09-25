//! xAI agent completion smoke test.

use rig::providers::xai;
use rig::wire::Wire as _;

use super::support::with_xai_cassette;
use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
async fn completion_smoke() {
    with_xai_cassette("agent/completion_smoke", |client| async move {
        let agent =
            rig::AgentBuilder::new(client.completion(xai::GROK_3_MINI).on(rig::transport()))
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
