//! OpenAI agent completion smoke test.

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::openai;

use crate::support::{PREAMBLE, PROMPT, assert_nontrivial_response};

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn completion_smoke() {
    let client = openai::Client::from_env();
    let agent = client.agent(openai::GPT_4O).preamble(PREAMBLE).build();

    let response = agent
        .prompt(PROMPT)
        .await
        .expect("completion should succeed");

    assert_nontrivial_response(&response);
}
