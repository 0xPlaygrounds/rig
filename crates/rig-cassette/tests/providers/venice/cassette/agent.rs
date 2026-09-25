//! Cassette-backed Venice completion coverage.

use super::super::{DEFAULT_MODEL, support::with_venice_cassette};
use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};
use rig::wire::Wire as _;

#[tokio::test]
async fn completion_smoke() {
    with_venice_cassette("agent/completion_smoke", |client| async move {
        let agent = rig::AgentBuilder::new(client.completion(DEFAULT_MODEL).on(rig::transport()))
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
