//! llama.cpp streaming coverage, including the migrated example path.

use rig::prelude::*;

use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response, collect_stream_final_response,
};

use super::support;

#[tokio::test]
#[ignore = "requires a local llama.cpp OpenAI-compatible server"]
async fn streaming_smoke() {
    let client = support::completions_client();
    let agent = client
        .agent(support::model_name())
        .preamble(STREAMING_PREAMBLE)
        .build();

    let mut stream = Box::pin(agent.runner(STREAMING_PROMPT).stream_run());
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming prompt should succeed");

    assert_nonempty_response(&response);
}

#[tokio::test]
#[ignore = "requires a local llama.cpp OpenAI-compatible server"]
async fn example_streaming_prompt() {
    let client = support::completions_client();
    let agent = client
        .agent(support::model_name())
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
