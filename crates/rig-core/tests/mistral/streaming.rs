//! Mistral streaming coverage, including the migrated example path.

use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::providers::mistral;
use rig_core::streaming::StreamingPrompt;

use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response, collect_stream_final_response,
};

use super::DEFAULT_MODEL;

#[tokio::test]
#[ignore = "requires MISTRAL_API_KEY"]
async fn streaming_smoke() {
    let client = mistral::Client::from_env().expect("client should build");
    let agent = client
        .agent(DEFAULT_MODEL)
        .preamble(STREAMING_PREAMBLE)
        .build();

    let mut stream = agent.stream_prompt(STREAMING_PROMPT).await;
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming prompt should succeed");

    assert_nonempty_response(&response);
}

#[tokio::test]
#[ignore = "requires MISTRAL_API_KEY"]
async fn example_streaming_prompt() {
    let client = mistral::Client::from_env().expect("client should build");
    let agent = client
        .agent(DEFAULT_MODEL)
        .preamble("Be precise and concise.")
        .temperature(0.5)
        .build();

    let mut stream = agent
        .stream_prompt("When and where and what type is the next solar eclipse?")
        .await;
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming prompt should succeed");

    assert_nonempty_response(&response);
}
