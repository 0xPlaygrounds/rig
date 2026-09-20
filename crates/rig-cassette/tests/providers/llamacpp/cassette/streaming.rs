//! llama.cpp streaming coverage, including the migrated example path.

use rig::prelude::*;

use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response, collect_stream_final_response,
};

use super::super::cassette_support::*;

#[rig_test_support::cassette(rig_test_support::recording::Scenario::live(
    "llamacpp/streaming/streaming_smoke"
))]
#[tokio::test]
async fn streaming_smoke() {
    with_llamacpp_cassette("streaming/streaming_smoke", |client| async move {
        let agent = client
            .agent(CASSETTE_MODEL)
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

#[rig_test_support::cassette(rig_test_support::recording::Scenario::live(
    "llamacpp/streaming/example_streaming_prompt"
))]
#[tokio::test]
async fn example_streaming_prompt() {
    with_llamacpp_cassette("streaming/example_streaming_prompt", |client| async move {
        let agent = client
            .agent(CASSETTE_MODEL)
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
