//! Cassette-backed OpenRouter agent completion smoke test.

use rig::prelude::*;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

use super::super::{DEFAULT_MODEL, support::with_openrouter_cassette};

#[tokio::test]
async fn completion_smoke() {
    with_openrouter_cassette("agent/completion_smoke", |client| async move {
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
