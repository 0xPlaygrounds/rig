//! xAI agent completion smoke test.

use rig::prelude::*;
use rig::providers::xai;

use super::support::with_xai_cassette;
use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[rig_test_support::cassette(rig_test_support::recording::Scenario::live(
    "xai/agent/completion_smoke"
))]
#[tokio::test]
async fn completion_smoke() {
    with_xai_cassette("agent/completion_smoke", |client| async move {
        let agent = client
            .agent(xai::GROK_3_MINI)
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
