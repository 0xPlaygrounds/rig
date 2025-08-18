use rig::prelude::*;
use std::env;

use rig::agent::stream_to_stdout;
use rig::{
    providers::huggingface::{self},
    streaming::StreamingPrompt,
};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create streaming agent with a single context prompt
    let api_key = &env::var("HUGGINGFACE_API_KEY").expect("HUGGINGFACE_API_KEY not set");

    println!("\n\nRunning Llama 3.1\n\n");
    hf_inference(api_key).await?;

    println!("\n\nRunning Deepseek R-1\n\n");
    together(api_key).await?;

    Ok(())
}

async fn hf_inference(api_key: &str) -> Result<(), anyhow::Error> {
    let agent = huggingface::Client::new(api_key)
        .agent("meta-llama/Llama-3.1-8B-Instruct")
        .preamble("Be precise and concise.")
        .temperature(0.5)
        .build();

    // Stream the response and print chunks as they arrive
    let mut stream = agent
        .stream_prompt("When and where and what type is the next solar eclipse?")
        .await;

    let _ = stream_to_stdout(&mut stream).await?;

    Ok(())
}

async fn together(api_key: &str) -> Result<(), anyhow::Error> {
    let agent = huggingface::ClientBuilder::new(api_key)
        .sub_provider(huggingface::SubProvider::Together)
        .build()?
        .agent("deepseek-ai/DeepSeek-R1")
        .preamble("Be precise and concise.")
        .temperature(0.5)
        .build();

    // Stream the response and print chunks as they arrive
    let mut stream = agent
        .stream_prompt("When and where and what type is the next solar eclipse?")
        .await;

    let _ = stream_to_stdout(&mut stream).await?;

    Ok(())
}
