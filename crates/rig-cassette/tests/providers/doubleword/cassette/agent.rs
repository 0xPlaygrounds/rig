//! Cassette-backed Doubleword completion coverage.

use rig::prelude::*;

use super::super::{DEFAULT_MODEL, support::with_doubleword_cassette};
use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
async fn completion_smoke() {
    with_doubleword_cassette("agent/completion_smoke", |client| async move {
        let agent = client
            .endpoint(|provider_config| provider_config.completion(DEFAULT_MODEL))
            .into_agent_builder()
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
