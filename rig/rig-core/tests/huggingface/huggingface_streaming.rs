//! Migrated from `examples/huggingface_streaming.rs`.

use rig::client::CompletionClient;
use rig::providers::huggingface::{self, SubProvider};
use rig::streaming::StreamingPrompt;

use crate::support::{assert_nonempty_response, collect_stream_final_response};

#[tokio::test]
#[ignore = "requires HUGGINGFACE_API_KEY"]
async fn default_inference_streaming() {
    let api_key = std::env::var("HUGGINGFACE_API_KEY").expect("HUGGINGFACE_API_KEY must be set");
    let agent = huggingface::Client::new(&api_key)
        .expect("client should build")
        .agent("meta-llama/Meta-Llama-3.1-8B-Instruct")
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
