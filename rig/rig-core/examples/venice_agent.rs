//! This example shows how to use Venice's OpenAI-compatible Chat Completions API with Rig.
//! Venice documents a `/chat/completions` endpoint, so this example uses
//! `openai::CompletionsClient` rather than Rig's default OpenAI Responses API client.

use rig::completion::Prompt;
use rig::prelude::*;
use rig::providers::openai;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let venice_api_key = std::env::var("VENICE_API_KEY").expect("VENICE_API_KEY not set");

    let agent = openai::CompletionsClient::builder()
        .api_key(&venice_api_key)
        .base_url("https://api.venice.ai/api/v1")
        .build()?
        .agent("venice-uncensored")
        .preamble("You are a helpful assistant")
        .build();

    let response = agent
        .prompt("Explain why OpenAI-compatible APIs are useful.")
        .await?;

    println!("Venice: {response}");

    Ok(())
}
