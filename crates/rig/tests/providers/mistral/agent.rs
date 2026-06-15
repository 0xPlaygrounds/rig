//! Mistral agent completion smoke test.

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::mistral;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

use super::DEFAULT_MODEL;

#[tokio::test]
#[ignore = "requires MISTRAL_API_KEY"]
async fn completion_smoke() {
    let client = mistral::Client::from_env().expect("client should build");
    let agent = client.agent(DEFAULT_MODEL).preamble(BASIC_PREAMBLE).build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("completion should succeed");

    assert_nonempty_response(&response);
}
