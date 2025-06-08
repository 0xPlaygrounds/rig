use anyhow::{anyhow, Result};
use rig::completion::Completion;
use rig::prelude::*;
use rig::providers::openrouter::Message;

use rig::providers;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt().init();

    tracing::info!("Trying to prompt an agent with reasoning effort added...");

    let reasoning = json!({
        "effort": "medium"
    });

    // Create agent with a single context prompt and two tools
    let calculator_agent = providers::openrouter::Client::from_env()
        .agent("google/gemini-2.5-flash-preview-05-20:thinking")
        .preamble("You are a helpful assistant.")
        .max_tokens(1024)
        .additional_params(reasoning)
        .build();

    tracing::info!("Hello, Gemini!");

    let response = calculator_agent
        .completion("Hello, Gemini!", Vec::new())
        .await
        .unwrap()
        .send()
        .await
        .unwrap()
        .raw_response;

    let Message::Assistant {
        content, reasoning, ..
    } = response.choices.into_iter().next().unwrap().message
    else {
        return Err(anyhow!(
            "Attempted to get first message in the response but it wasn't an assistant message"
        ));
    };

    tracing::info!("Content: {content:?}");

    let Some(reasoning) = reasoning else {
        return Err(anyhow!("No reasoning tokens were used"));
    };

    tracing::info!("Reasoning: {reasoning}");

    Ok(())
}
