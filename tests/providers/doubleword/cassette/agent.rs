//! Cassette-backed Doubleword completion coverage.

use rig::completion::Prompt;
use rig::prelude::AgentClientExt;

use super::super::{DEFAULT_MODEL, support::with_doubleword_cassette};
use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
async fn completion_smoke() {
    with_doubleword_cassette("agent/completion_smoke", |client| async move {
        let agent = client.agent(DEFAULT_MODEL).preamble(BASIC_PREAMBLE).build();
        let response = agent
            .prompt(BASIC_PROMPT)
            .await
            .expect("completion should succeed");
        assert_nonempty_response(&response);
    })
    .await;
}
