//! Demonstrates host-managed conversation memory with an in-memory store.
//!
//! Memory is not an agent slot: the host loads prior history before a prompt
//! and appends the finished turn afterwards, keyed by a `conversation_id`.
//! The same agent serves any number of conversations because the id lives at
//! the call site, not on the agent.
//!
//! The recipe below mirrors the semantics Rig used to apply internally: a
//! load failure is fatal (the run never starts), an append failure warns and
//! proceeds (the response is still surfaced), and passing your own history
//! simply means you skip both calls.
//!
//! Requires `OPENAI_API_KEY`.

use anyhow::Result;
use rig::agent::Agent;
use rig::memory::InMemoryConversationMemory;
use rig::prelude::*;
use rig::providers::openai;

/// One turn: load-before, run, append-after.
async fn ask(
    agent: &Agent,
    memory: &InMemoryConversationMemory,
    conversation_id: &str,
    prompt: &str,
) -> Result<String> {
    // Load-before: a load failure is fatal, so the run never starts.
    let history = memory.load(conversation_id)?;

    let response = agent.runner(prompt).history(history).run().await?;

    // Append-after: warn and proceed, so a store hiccup never drops a reply.
    if let Some(messages) = &response.messages
        && let Err(error) = memory.append(conversation_id, messages.clone())
    {
        tracing::warn!(
            %error,
            conversation_id,
            "conversation memory append failed; surfacing final response anyway"
        );
    }

    Ok(response.output)
}

#[tokio::main]
async fn main() -> Result<()> {
    // For named history-shaping policies (sliding window, token budget,
    // rolling summaries), depend on the `rig-memory` companion crate and use
    // its `PolicyMemory`. Here we use the bare store.
    let memory = InMemoryConversationMemory::new();

    let cfg = openai::functions::Config::from_env(openai::GPT_4O)?;
    let agent = AgentBuilder::new(ProviderConfig::OpenAi(cfg))
        .preamble("You are a helpful assistant with persistent memory.")
        .build();

    let first = ask(&agent, &memory, "user-123", "My name is Alice.").await?;
    println!("turn 1: {first}");

    let second = ask(&agent, &memory, "user-123", "What's my name?").await?;
    println!("turn 2: {second}");

    Ok(())
}
