//! Cassette-backed OpenRouter streaming coverage.

use rig::prelude::*;

use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response, collect_stream_final_response,
};

use super::super::{DEFAULT_MODEL, support::with_openrouter_cassette};

#[tokio::test]
async fn streaming_smoke() {
    with_openrouter_cassette("streaming/streaming_smoke", |client| async move {
        let agent = client
            .endpoint(|provider_config| provider_config.completion(DEFAULT_MODEL))
            .into_agent_builder()
            .preamble(STREAMING_PREAMBLE)
            .build();

        let mut stream = agent.prompt(STREAMING_PROMPT).stream();
        let response = collect_stream_final_response(&mut stream)
            .await
            .expect("streaming prompt should succeed");

        assert_nonempty_response(&response);
    })
    .await;
}

#[tokio::test]
async fn example_streaming_prompt() {
    with_openrouter_cassette("streaming/example_streaming_prompt", |client| async move {
        let agent = client
            .endpoint(|provider_config| provider_config.completion(DEFAULT_MODEL))
            .into_agent_builder()
            .preamble("Be precise and concise.")
            .temperature(0.5)
            .build();

        let mut stream = agent
            .prompt("When and where and what type is the next solar eclipse?")
            .stream();
        let response = collect_stream_final_response(&mut stream)
            .await
            .expect("streaming prompt should succeed");

        assert_nonempty_response(&response);
    })
    .await;
}
