//! OpenAI streaming coverage, including the migrated example path.

use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::providers::openai;
use rig_core::streaming::StreamingPrompt;

use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response, collect_stream_final_response,
};

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn streaming_smoke() {
    let client = openai::Client::from_env().expect("client should build");
    let agent = client
        .agent(openai::GPT_4O)
        .preamble(STREAMING_PREAMBLE)
        .build();

    let mut stream = agent.stream_prompt(STREAMING_PROMPT).await;
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming prompt should succeed");

    assert_nonempty_response(&response);
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn example_streaming_prompt() {
    let client = openai::Client::from_env().expect("client should build");
    let agent = client
        .agent(openai::GPT_4O)
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
