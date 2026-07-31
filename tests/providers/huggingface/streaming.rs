//! Hugging Face streaming coverage for the default and Together-backed inference paths.

use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response, collect_stream_final_response,
};
use rig::prelude::*;
use rig::providers::huggingface::{self, SubProvider};

#[tokio::test]
#[ignore = "requires HUGGINGFACE_API_KEY"]
async fn streaming_smoke() {
    let cfg = huggingface::functions::Config::from_env("meta-llama/Meta-Llama-3.1-8B-Instruct")
        .expect("config should build");
    let agent = AgentBuilder::new(ProviderConfig::HuggingFace(cfg))
        .preamble(STREAMING_PREAMBLE)
        .build();

    let mut stream = agent.runner(STREAMING_PROMPT).stream_run();
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming prompt should succeed");

    assert_nonempty_response(&response);
}

#[tokio::test]
#[ignore = "requires HUGGINGFACE_API_KEY"]
async fn together_subprovider_streaming() {
    let api_key = std::env::var("HUGGINGFACE_API_KEY").expect("HUGGINGFACE_API_KEY must be set");
    let cfg = huggingface::functions::Config::new("deepseek-ai/DeepSeek-R1")
        .with_api_key(&api_key)
        .with_sub_provider(SubProvider::Together);
    let agent = AgentBuilder::new(ProviderConfig::HuggingFace(cfg))
        .preamble("Be precise and concise.")
        .temperature(0.5)
        .build();

    let mut stream = agent
        .runner("When and where and what type is the next solar eclipse?")
        .stream_run();
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming prompt should succeed");

    assert_nonempty_response(&response);
}
