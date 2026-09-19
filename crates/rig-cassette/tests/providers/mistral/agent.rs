//! Mistral agent completion smoke test.

use rig::prelude::*;
use rig::providers::openai::wire::{MISTRAL, OpenAI};

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

use super::DEFAULT_MODEL;

#[tokio::test]
#[ignore = "requires MISTRAL_API_KEY"]
async fn completion_smoke() {
    let client = OpenAI::from_env_with(&MISTRAL)
        .expect("MISTRAL_API_KEY should be set")
        .bound()
        .expect("client should build");
    let agent = client.agent(DEFAULT_MODEL).preamble(BASIC_PREAMBLE).build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("completion should succeed");

    assert_nonempty_response(&response.output);
}
