//! Hugging Face streaming coverage for the default and Together-backed inference paths.

use rig::prelude::*;
use rig::providers::openai::wire::{HUGGINGFACE, OpenAI, SubRoute};

use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response, collect_stream_final_response,
};

#[tokio::test]
#[ignore = "requires HUGGINGFACE_API_KEY"]
async fn streaming_smoke() {
    let provider = OpenAI::from_env_with(&HUGGINGFACE)
        .expect("config should build from env")
        .bound()
        .expect("transport should build");
    let agent = provider
        .agent("meta-llama/Meta-Llama-3.1-8B-Instruct")
        .preamble(STREAMING_PREAMBLE)
        .build();

    let mut stream = agent.prompt(STREAMING_PROMPT).stream();
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming prompt should succeed");

    assert_nonempty_response(&response);
}

#[tokio::test]
#[ignore = "requires HUGGINGFACE_API_KEY"]
async fn together_subprovider_streaming() {
    let agent = OpenAI::from_env_with(&HUGGINGFACE)
        .expect("config should build from env")
        .with_sub_route(SubRoute::Together)
        .bound()
        .expect("transport should build")
        .agent("deepseek-ai/DeepSeek-R1")
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
