//! Hugging Face streaming coverage for the default and Together-backed inference paths.

use rig::client::{CompletionClient, ProviderClient};
use rig::providers::huggingface::{self, SubProvider};
use rig::streaming::StreamingPrompt;

use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response, collect_stream_final_response,
};

#[tokio::test]
#[ignore = "requires HUGGINGFACE_API_KEY"]
async fn streaming_smoke() {
    let client = huggingface::Client::from_env();
    let agent = client
        .agent("meta-llama/Meta-Llama-3.1-8B-Instruct")
        .preamble(STREAMING_PREAMBLE)
        .build();

    let mut stream = agent.stream_prompt(STREAMING_PROMPT).await;
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming prompt should succeed");

    assert_nonempty_response(&response);
}

#[tokio::test]
#[ignore = "requires HUGGINGFACE_API_KEY"]
async fn together_subprovider_streaming() {
    let api_key = std::env::var("HUGGINGFACE_API_KEY").expect("HUGGINGFACE_API_KEY must be set");
    let agent = huggingface::Client::builder()
        .api_key(&api_key)
        .subprovider(SubProvider::Together)
        .build()
        .expect("client should build")
        .agent("deepseek-ai/DeepSeek-R1")
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
