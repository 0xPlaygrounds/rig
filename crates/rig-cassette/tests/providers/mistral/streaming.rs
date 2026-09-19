//! Mistral streaming coverage, including the migrated example path.

use rig::prelude::*;
use rig::providers::openai::wire::{MISTRAL, OpenAI};

use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response, collect_stream_final_response,
};

use super::DEFAULT_MODEL;

#[tokio::test]
#[ignore = "requires MISTRAL_API_KEY"]
async fn streaming_smoke() {
    let client = OpenAI::from_env_with(&MISTRAL)
        .expect("MISTRAL_API_KEY should be set")
        .bound()
        .expect("client should build");
    let agent = client
        .agent(DEFAULT_MODEL)
        .preamble(STREAMING_PREAMBLE)
        .build();

    let mut stream = agent.prompt(STREAMING_PROMPT).stream();
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming prompt should succeed");

    assert_nonempty_response(&response);
}

#[tokio::test]
#[ignore = "requires MISTRAL_API_KEY"]
async fn example_streaming_prompt() {
    let client = OpenAI::from_env_with(&MISTRAL)
        .expect("MISTRAL_API_KEY should be set")
        .bound()
        .expect("client should build");
    let agent = client
        .agent(DEFAULT_MODEL)
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
}
