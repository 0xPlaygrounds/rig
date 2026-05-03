//! Demonstrates Rig-managed conversation memory with streaming.
//!
//! The agent loads prior history before each prompt and appends the new turn
//! after the streaming response completes, identified by a `conversation_id`.
//!
//! Requires `OPENAI_API_KEY`.

use anyhow::{Result, anyhow};
use futures::StreamExt;
use rig::agent::{MultiTurnStreamItem, StreamingResult};
use rig::client::{CompletionClient, ProviderClient};
use rig::memory::InMemoryConversationMemory;
use rig::providers::openai;
use rig::streaming::StreamingPrompt;

async fn collect_final<R>(stream: &mut StreamingResult<R>) -> Result<String> {
    let mut final_response = None;
    while let Some(item) = stream.next().await {
        if let MultiTurnStreamItem::FinalResponse(response) = item? {
            final_response = Some(response.response().to_owned());
        }
    }
    final_response.ok_or_else(|| anyhow!("stream finished without a final response"))
}

#[tokio::main]
async fn main() -> Result<()> {
    let memory = InMemoryConversationMemory::new();

    let agent = openai::Client::from_env()?
        .agent(openai::GPT_4O)
        .preamble("You are a helpful assistant with persistent memory.")
        .memory(memory)
        .build();

    let mut first = agent
        .stream_prompt("My name is Alice.")
        .conversation("user-123")
        .await;
    let reply1 = collect_final(&mut first).await?;
    println!("turn 1: {reply1}");

    let mut second = agent
        .stream_prompt("What's my name?")
        .conversation("user-123")
        .await;
    let reply2 = collect_final(&mut second).await?;
    println!("turn 2: {reply2}");

    Ok(())
}
