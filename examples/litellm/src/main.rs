//! Runs a Rig agent through LiteLLM's OpenAI-compatible Chat Completions API.
//! Requires a running proxy and `LITELLM_API_KEY`; see the adjacent README
//! for setup and the optional `LITELLM_BASE_URL` and `LITELLM_MODEL` overrides.

use anyhow::{Context, Result};
use rig::prelude::*;
use rig::providers::openai::OpenAIConfig;

#[tokio::main]
async fn main() -> Result<()> {
    let api_key = std::env::var("LITELLM_API_KEY").context(
        "Set LITELLM_API_KEY to your proxy key (or 'unused' for an unauthenticated proxy)",
    )?;
    let base_url =
        std::env::var("LITELLM_BASE_URL").unwrap_or_else(|_| "http://localhost:4000/v1".to_owned());
    let model = std::env::var("LITELLM_MODEL").unwrap_or_else(|_| "rig-chat".to_owned());

    let client = OpenAIConfig::new(api_key).with_base_url(base_url).client();
    // Select Chat Completions explicitly; OpenAI's default route is Responses.
    let agent = AgentBuilder::new(client.chat(model))
        .preamble("You are a helpful assistant. Keep your answers concise.")
        .build();

    let response = agent
        .prompt("Explain what an LLM gateway does in one sentence.")
        .await?;
    println!("{}", response.output());

    Ok(())
}
