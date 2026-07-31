//! Demonstrates host-managed conversation memory with streaming.
//!
//! Same recipe as the blocking example: the host loads history before the
//! prompt and appends the finished turn once the stream has produced its
//! final response. The final response carries the run's committed transcript
//! (`PromptResponse::messages`), which is exactly what gets appended.
//!
//! Requires `OPENAI_API_KEY`.

use anyhow::{Result, anyhow};
use futures::StreamExt;
use rig::agent::Agent;
use rig::completion::Message;
use rig::memory::InMemoryConversationMemory;
use rig::prelude::*;
use rig::providers::openai;
use rig::stream::AgentStreamItem;

/// One streamed turn: load-before, stream, append-after.
async fn ask(
    agent: &Agent,
    memory: &InMemoryConversationMemory,
    conversation_id: &str,
    prompt: &str,
) -> Result<String> {
    // Load-before: a load failure is fatal, so the run never starts.
    let history = memory.load(conversation_id)?;

    let mut stream = Box::pin(agent.runner(prompt).history(history).stream_run());

    let mut output = None;
    let mut committed: Vec<Message> = Vec::new();
    while let Some(item) = stream.next().await {
        if let AgentStreamItem::Final(response) = item? {
            output = Some(response.output().to_owned());
            committed = response.messages().unwrap_or_default().to_vec();
        }
    }
    let output = output.ok_or_else(|| anyhow!("stream finished without a final response"))?;

    // Append-after: warn and proceed, so a store hiccup never drops a reply.
    if !committed.is_empty()
        && let Err(error) = memory.append(conversation_id, committed)
    {
        tracing::warn!(
            %error,
            conversation_id,
            "conversation memory append failed; surfacing final response anyway"
        );
    }

    Ok(output)
}

#[tokio::main]
async fn main() -> Result<()> {
    let memory = InMemoryConversationMemory::new();

    let cfg = openai::functions::Config::from_env(openai::GPT_4O)?;
    let agent = AgentBuilder::new(cfg)
        .preamble("You are a helpful assistant with persistent memory.")
        .build();

    let first = ask(&agent, &memory, "user-123", "My name is Alice.").await?;
    println!("turn 1: {first}");

    let second = ask(&agent, &memory, "user-123", "What's my name?").await?;
    println!("turn 2: {second}");

    Ok(())
}
