//! Deferred tools + the built-in `tool_search` meta-tool — the answer to "more
//! tools than fit in a prompt".
//!
//! Deferred tools are registered with `.deferred_tool(...)`: they are executable
//! but **withheld** from the advertised tool set (zero schema tokens, zero
//! `definition()` calls) until the model discovers them by calling the built-in
//! `tool_search` tool with a capability query. Once a `tool_search` result names
//! a tool, it is advertised on subsequent turns and dispatched like any other
//! tool. Reveal state is reconstructed from the conversation transcript, so runs
//! stay stateless/resumable and the request prefix stays prompt-cache friendly.
//!
//! This is the pattern Pydantic AI (`defer_loading` + `ToolSearch`), the OpenAI
//! Agents SDK (`defer_loading` + `ToolSearchTool`), and this Claude Code harness
//! all converge on. Swap the default keyword search for embeddings/BM25 with
//! `ToolServer::tool_search_fn` when you build the server directly.
//!
//! Requires `OPENAI_API_KEY`. Run with: `cargo run -p deferred_tools`

use anyhow::Result;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{Prompt, ToolDefinition};
use rig::providers::openai;
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::json;

#[derive(Debug, thiserror::Error)]
#[error("tool error")]
struct ToolFail;

#[derive(Deserialize)]
struct BinaryArgs {
    x: i64,
    y: i64,
}

/// A deferred "add" tool.
struct Add;
impl Tool for Add {
    const NAME: &'static str = "add";
    type Error = ToolFail;
    type Args = BinaryArgs;
    type Output = i64;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Add two integers together (sum, plus).".to_string(),
            parameters: json!({
                "type": "object",
                "properties": { "x": { "type": "integer" }, "y": { "type": "integer" } },
                "required": ["x", "y"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(args.x + args.y)
    }
}

/// A deferred "subtract" tool.
struct Subtract;
impl Tool for Subtract {
    const NAME: &'static str = "subtract";
    type Error = ToolFail;
    type Args = BinaryArgs;
    type Output = i64;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Subtract the second integer from the first (difference, minus)."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": { "x": { "type": "integer" }, "y": { "type": "integer" } },
                "required": ["x", "y"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(args.x - args.y)
    }
}

#[derive(Deserialize)]
struct CityArgs {
    city: String,
}

/// A deferred "weather" tool (stub).
struct Weather;
impl Tool for Weather {
    const NAME: &'static str = "weather";
    type Error = ToolFail;
    type Args = CityArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Look up the current weather forecast for a city.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(format!("It is sunny and 21°C in {}.", args.city))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // None of these tool schemas are sent to the model up front — only
    // `tool_search` is advertised. The model searches, then calls the match.
    let agent = openai::Client::from_env()?
        .agent(openai::GPT_4O)
        .preamble(
            "You have access to a large toolbox, but the tools are hidden. When you need a \
             capability, call `tool_search` with a short query to discover the relevant tool, \
             then call that tool by name.",
        )
        .deferred_tool(Add)
        .deferred_tool(Subtract)
        .deferred_tool(Weather)
        .build();

    // Expect: tool_search({queries:["add", ...]}) -> reveals `add` -> add(21, 21) -> 42.
    let answer = agent.prompt("What is 21 plus 21?").max_turns(6).await?;
    println!("{answer}");
    Ok(())
}
