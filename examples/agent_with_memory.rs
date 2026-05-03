//! Demonstrates Rig-managed conversation memory with an in-memory backend.
//!
//! The agent loads prior history before each prompt and appends the new turn
//! after a successful response, identified by a `conversation_id`. Reuses the
//! same agent across multiple conversations by passing the id per-request.
//!
//! Requires `OPENAI_API_KEY`.

use anyhow::Result;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::memory::InMemoryConversationMemory;
use rig::providers::openai;

#[tokio::main]
async fn main() -> Result<()> {
    // For named history-shaping policies (sliding window, token budget, etc.),
    // depend on the `rig-memory` companion crate. Here we use the bare backend.
    let memory = InMemoryConversationMemory::new();

    let agent = openai::Client::from_env()?
        .agent(openai::GPT_4O)
        .preamble("You are a helpful assistant with persistent memory.")
        .memory(memory)
        .build();

    let first = agent
        .prompt("My name is Alice.")
        .conversation("user-123")
        .await?;
    println!("turn 1: {first}");

    let second = agent
        .prompt("What's my name?")
        .conversation("user-123")
        .await?;
    println!("turn 2: {second}");

    Ok(())
}
