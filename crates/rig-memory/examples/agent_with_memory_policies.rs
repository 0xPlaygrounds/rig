//! Demonstrates `rig-memory` history-shaping policies on top of a Rig agent.
//!
//! Two backends are configured against the same prompt:
//!
//! * `SlidingWindowMemory` — keeps the most recent fixed number of messages.
//! * `TokenWindowMemory` — keeps the most recent messages that fit within a
//!   token budget supplied by a [`TokenCounter`].
//!
//! Both policies are converted into a `MessageFilter` via
//! [`IntoFilter::into_filter`] and attached to an
//! [`InMemoryConversationMemory`] backend with `with_filter`.
//!
//! Requires `OPENAI_API_KEY`.

use anyhow::Result;
use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::completion::{Message, Prompt};
use rig_core::providers::openai;
use rig_memory::{InMemoryConversationMemory, IntoFilter, SlidingWindowMemory, TokenWindowMemory};

fn approx_token_count(message: &Message) -> usize {
    let text = match message {
        Message::User { content, .. } => content
            .iter()
            .filter_map(|c| match c {
                rig_core::completion::message::UserContent::Text(t) => Some(t.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" "),
        Message::Assistant { content, .. } => content
            .iter()
            .filter_map(|c| match c {
                rig_core::completion::message::AssistantContent::Text(t) => Some(t.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" "),
        Message::System { content } => content.clone(),
    };
    text.split_whitespace().count().max(1)
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = openai::Client::from_env()?;

    let sliding_memory = InMemoryConversationMemory::new()
        .with_filter(SlidingWindowMemory::last_messages(20).into_filter());

    let sliding_agent = client
        .agent(openai::GPT_4O)
        .preamble("You are a helpful assistant. Keep responses short.")
        .memory(sliding_memory)
        .build();

    let reply = sliding_agent
        .prompt("Remember: my favorite color is teal.")
        .conversation("alice")
        .await?;
    println!("[sliding] {reply}");

    let token_memory = InMemoryConversationMemory::new()
        .with_filter(TokenWindowMemory::new(256, approx_token_count).into_filter());

    let token_agent = client
        .agent(openai::GPT_4O)
        .preamble("You are a helpful assistant. Keep responses short.")
        .memory(token_memory)
        .build();

    let reply = token_agent
        .prompt("Plan a 3-day trip to Kyoto.")
        .conversation("alice")
        .await?;
    println!("[token]   {reply}");

    Ok(())
}
