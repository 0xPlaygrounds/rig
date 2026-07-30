//! Demonstrates `rig-memory` history-shaping policies around a Rig agent.
//!
//! Two policies are exercised against the same host recipe:
//!
//! * `MemoryPolicy::SlidingWindow` — keeps the most recent fixed number of
//!   messages.
//! * `MemoryPolicy::TokenWindow` — keeps the most recent messages that fit
//!   within a token budget, counted by a `TokenCounter`.
//!
//! Policies are data, not callbacks: `PolicyMemory::load` returns the shaped
//! history and `PolicyMemory::append` hands back an `AppendOutcome` naming
//! what fell out of the active window, so the host decides what to archive
//! and when to compact.
//!
//! Requires `OPENAI_API_KEY`.

use anyhow::Result;
use rig_agent::agent::Agent;
use rig_agent::prelude::*;
use rig_core::providers::openai;
use rig_memory::{
    Compactor, HeuristicTokenCounter, InMemoryConversationMemory, MemoryPolicy, PolicyMemory,
    TokenCounter,
};

/// One turn: load-before, run, append-after, then act on the outcome.
async fn ask(
    agent: &Agent,
    memory: &PolicyMemory,
    conversation_id: &str,
    prompt: &str,
) -> Result<String> {
    // Load-before: the policy-shaped history (plus any rolling summary).
    let history = memory.load(conversation_id)?;

    let response = agent.runner(prompt).history(history).run().await?;

    // Append-after: warn and proceed on failure, then act on the outcome.
    if let Some(messages) = &response.messages {
        match memory.append(conversation_id, messages.clone()) {
            Ok(outcome) => {
                for demoted in &outcome.demoted {
                    // Archive into a long-tail store (vector RAG, episodic
                    // recall, ...) — nothing is lost when the window slides.
                    tracing::debug!(?demoted, "message demoted out of the active window");
                }
                if let Some(request) = &outcome.compaction {
                    // A compactor is configured: fold the evicted prefix into
                    // the conversation's rolling summary.
                    memory.compact(request);
                }
            }
            Err(error) => tracing::warn!(
                %error,
                conversation_id,
                "conversation memory append failed; surfacing final response anyway"
            ),
        }
    }

    Ok(response.output)
}

#[tokio::main]
async fn main() -> Result<()> {
    // Providers are plain configuration: no client object, just a
    // `ProviderConfig` arm wrapping `openai::functions::Config`.
    let agent = AgentBuilder::new(ProviderConfig::OpenAi(openai::functions::Config::from_env(
        openai::GPT_4O,
    )?))
    .preamble("You are a helpful assistant. Keep responses short.")
    .build();

    // Sliding window: keep the last 20 messages, roll the rest into a summary.
    let sliding_memory = PolicyMemory::new(
        InMemoryConversationMemory::new(),
        MemoryPolicy::sliding_window(20),
    )
    .with_compactor(Compactor::template());

    let reply = ask(
        &agent,
        &sliding_memory,
        "alice",
        "Remember: my favorite color is teal.",
    )
    .await?;
    println!("[sliding] {reply}");

    // Token window: keep whatever fits in ~256 tokens of recent history.
    let token_memory = PolicyMemory::new(
        InMemoryConversationMemory::new(),
        MemoryPolicy::token_window(
            256,
            TokenCounter::Heuristic(HeuristicTokenCounter::openai()),
        ),
    );

    let reply = ask(
        &agent,
        &token_memory,
        "alice",
        "Plan a 3-day trip to Kyoto.",
    )
    .await?;
    println!("[token]   {reply}");

    Ok(())
}
