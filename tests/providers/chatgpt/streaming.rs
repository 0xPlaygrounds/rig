//! ChatGPT streaming smoke tests.

use rig::prelude::*;

use crate::chatgpt::{LIVE_MODEL, live_client};
use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response, collect_stream_final_response,
};

#[tokio::test]
#[ignore = "requires ChatGPT credentials or existing OAuth cache"]
async fn streaming_smoke() {
    let agent = live_client()
        .agent(LIVE_MODEL)
        .preamble(STREAMING_PREAMBLE)
        .build();

    let mut stream = Box::pin(agent.runner(STREAMING_PROMPT).stream_run());
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("ChatGPT stream should succeed");

    assert_nonempty_response(&response);
}

#[tokio::test]
#[ignore = "requires ChatGPT credentials or existing OAuth cache"]
async fn example_streaming_prompt() {
    let agent = live_client()
        .agent(LIVE_MODEL)
        .preamble("Be precise and concise.")
        .temperature(0.5)
        .build();

    let mut stream = Box::pin(
        agent
            .runner("When and where and what type is the next solar eclipse?")
            .stream_run(),
    );
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming prompt should succeed");

    assert_nonempty_response(&response);
}
