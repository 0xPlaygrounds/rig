//! Mistral streaming coverage, including the migrated example path.

use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response, collect_stream_final_response,
};
use rig::prelude::*;
use rig::providers::mistral;

use super::DEFAULT_MODEL;

#[tokio::test]
#[ignore = "requires MISTRAL_API_KEY"]
async fn streaming_smoke() {
    let client = mistral::Client::from_env().expect("client should build");
    let agent = client
        .agent(DEFAULT_MODEL)
        .preamble(STREAMING_PREAMBLE)
        .build();

    let mut stream = Box::pin(agent.runner(STREAMING_PROMPT).stream_run());
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
