//! Shows how to inspect per-model-call usage in an agent stream.
//!
//! This is useful when an agent uses tools. The final `response.usage()` is
//! intentionally aggregated across all model calls in the agent run:
//!
//! - call 0: prompt + tool request
//! - call 1: prompt + prior tool call + tool result + final answer
//!
//! That aggregate is useful for total cost, but it does not tell you the final
//! prompt/context size. For that, inspect the last entry from
//! `response.per_call_usage()` and use `input_tokens`.
//!
//! Requires `OPENAI_API_KEY`.
//!
//! For OpenAI:
//! `cargo run --example openai_streaming_per_call_usage`
//!
//! For OpenAI-compatible servers, for example llama.cpp:
//! `OPENAI_BASE_URL=http://localhost:8080/v1 OPENAI_API_KEY=local OPENAI_MODEL=local-model cargo run --example openai_streaming_per_call_usage`

use anyhow::{Result, anyhow};
use futures::StreamExt;
use rig::agent::MultiTurnStreamItem;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{ToolDefinition, Usage};
use rig::providers::openai;
use rig::streaming::{StreamedAssistantContent, StreamingPrompt};
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::io::{self, Write};

#[derive(Debug, thiserror::Error)]
#[error("project status lookup failed")]
struct ProjectStatusError;

#[derive(Debug, Deserialize)]
struct ProjectStatusArgs {
    ticket: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct ProjectStatusTool;

impl Tool for ProjectStatusTool {
    const NAME: &'static str = "lookup_project_status";

    type Error = ProjectStatusError;
    type Args = ProjectStatusArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Look up the current status for an internal project ticket.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "ticket": {
                        "type": "string",
                        "description": "The internal project ticket to look up"
                    }
                },
                "required": ["ticket"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(format!(
            "{} is approved for release after the final usage metrics check.",
            args.ticket
        ))
    }
}

fn print_usage(label: &str, usage: Usage) {
    println!(
        "{label}: input_tokens={}, output_tokens={}, total_tokens={}",
        usage.input_tokens, usage.output_tokens, usage.total_tokens
    );
}

#[tokio::main]
async fn main() -> Result<()> {
    let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| openai::GPT_4O_MINI.to_string());

    let agent = openai::CompletionsClient::from_env()?
        .agent(model)
        .preamble(
            "You are a concise release assistant. The user will ask about an \
             internal ticket. Call `lookup_project_status` exactly once before \
             answering. After the tool result is available, answer directly and \
             do not call another tool.",
        )
        .max_tokens(512)
        .tool(ProjectStatusTool)
        .build();

    let mut stream = agent
        .stream_prompt("Check ticket RIG-usage-42 and summarize the result in one sentence.")
        .multi_turn(4)
        .await;

    let mut final_response = None;

    while let Some(item) = stream.next().await {
        match item? {
            MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(text)) => {
                print!("{}", text.text);
                io::stdout().flush()?;
            }
            MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::ToolCall {
                tool_call,
                ..
            }) => {
                println!("\n\nmodel requested tool: {}", tool_call.function.name);
            }
            MultiTurnStreamItem::StreamUserItem(_) => {
                println!("tool result sent back to model");
            }
            MultiTurnStreamItem::ModelCallUsage { call_index, usage } => {
                print_usage(&format!("model call {call_index} usage"), usage);
            }
            MultiTurnStreamItem::FinalResponse(response) => {
                final_response = Some(response);
            }
            _ => {}
        }
    }

    let response = final_response.ok_or_else(|| anyhow!("stream ended without final response"))?;

    println!("\n\nfinal response: {}", response.response());
    print_usage("aggregate agent usage", response.usage());

    if let Some(final_call_usage) = response.per_call_usage().last().copied() {
        print_usage("final model call usage", final_call_usage);
        println!(
            "final prompt/context token length: {}",
            final_call_usage.input_tokens
        );
    }

    Ok(())
}
