//! Migrated from `examples/agent_with_galadriel.rs`.

use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::providers::galadriel;

use crate::support::assert_nonempty_response;

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
