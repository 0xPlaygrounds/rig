//! Galadriel agent completion smoke test.

use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::completion::Prompt;
use rig_core::providers::galadriel;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires GALADRIEL_API_KEY"]
async fn completion_smoke() {
    let client = galadriel::Client::from_env().expect("galadriel client should build");
    let agent = client
        .agent(galadriel::GPT_4O)
        .preamble(BASIC_PREAMBLE)
        .build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("completion should succeed");

    assert_nonempty_response(&response);
}

#[tokio::test]
#[ignore = "requires GALADRIEL_API_KEY"]
async fn builder_completion_smoke() {
    let api_key = std::env::var("GALADRIEL_API_KEY").expect("GALADRIEL_API_KEY must be set");
    let fine_tune_api_key = std::env::var("GALADRIEL_FINE_TUNE_API_KEY").ok();

    let mut builder = galadriel::Client::builder().api_key(&api_key);
    if let Some(fine_tune_api_key) = fine_tune_api_key.as_deref() {
        builder = builder.fine_tune_api_key(fine_tune_api_key);
    }

    let client = builder.build().expect("client should build");
    let agent = client
        .agent(galadriel::GPT_4O)
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .build();

    let response = agent
        .prompt("Entertain me!")
        .await
        .expect("prompt should succeed");

    assert_nonempty_response(&response);
}
