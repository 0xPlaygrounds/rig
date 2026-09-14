//! llama.cpp agent completion smoke test.

use rig::prelude::*;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

use super::super::cassette_support::*;

#[tokio::test]
async fn completion_smoke() {
    with_llamacpp_cassette("agent/completion_smoke", |client| async move {
        let agent = client
            .agent(CASSETTE_MODEL)
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
