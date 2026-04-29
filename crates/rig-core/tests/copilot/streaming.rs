//! Copilot streaming coverage, including the migrated example path.

use rig_core::client::CompletionClient;
use rig_core::streaming::StreamingPrompt;

use crate::copilot::{LIVE_MODEL, live_client};
use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response, collect_stream_final_response,
};

#[tokio::test]
#[ignore = "requires Copilot credentials or existing OAuth cache"]
async fn streaming_smoke() {
    let agent = live_client()
        .agent(LIVE_MODEL)
        .preamble(STREAMING_PREAMBLE)
        .build();

    let mut stream = agent.stream_prompt(STREAMING_PROMPT).await;
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming prompt should succeed");

    assert_nonempty_response(&response);
}

#[tokio::test]
#[ignore = "requires Copilot credentials or existing OAuth cache"]
async fn example_streaming_prompt() {
    let agent = live_client()
        .agent(LIVE_MODEL)
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
