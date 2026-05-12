//! OpenAI agent completion smoke test.

use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::providers::openai;

use crate::cassettes::ProviderCassette;
use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
async fn completion_smoke() {
    let cassette = ProviderCassette::start(
        "openai",
        "agent/completion_smoke",
        "https://api.openai.com/v1",
    )
    .await;
    let client = openai::Client::builder()
        .api_key(cassette.api_key("OPENAI_API_KEY"))
        .base_url(cassette.base_url())
        .build()
        .expect("client should build");
    let agent = client
        .agent(openai::GPT_4O)
        .preamble(BASIC_PREAMBLE)
        .build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("completion should succeed");

    assert_nonempty_response(&response);
    cassette.finish().await;
}
