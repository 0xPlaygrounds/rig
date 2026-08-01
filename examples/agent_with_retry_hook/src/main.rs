//! Retries a completed, tool-free model turn from a hook record.
//!
//! Hooks are attach-and-forget records: [`retry_on_marker`] is a plain function
//! returning a [`HookEntry`] whose closure owns the policy limit and its own
//! attempt counter — a host-owned `Arc<AtomicUsize>` captured by the closure,
//! replacing the old run-scoped scratchpad. Rig does not add a separate retry
//! counter to the agent: every retry consumes the request's existing `max_turns`
//! model-call budget.
//!
//! [`RetryMode::Feedback`] preserves the rejected assistant response and adds a
//! corrective user message. [`RetryMode::Repeat`] discards the response and
//! reuses the same prompt and preceding history with fresh request preparation.
//! Completion-call hooks, retrieval, and dynamic tool resolution therefore run
//! again. Tool-bearing turns must instead be handled by tool-call hooks.
//!
//! Requires `OPENAI_API_KEY`. Run with:
//! `cargo run -p agent_with_retry_hook`.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use rig::agent::ModelTurnAction;
use rig::hooks::{HookDecision, HookEntry, HookEvent};
use rig::message::AssistantContent;
use rig::prelude::*;
use rig::providers::openai;

#[derive(Clone)]
enum RetryMode {
    Repeat,
    Feedback(String),
}

/// Rejects any tool-free model turn whose text contains `marker`, retrying up to
/// `max_retries` times according to `mode`.
fn retry_on_marker(marker: impl Into<String>, max_retries: usize, mode: RetryMode) -> HookEntry {
    // The attempt counter lives with the hook record, not in a run-scoped
    // scratchpad: the closure captures it and every clone of the entry shares it.
    let attempts = Arc::new(AtomicUsize::new(0));
    let marker = marker.into();
    HookEntry::sync("retry-on-marker", move |event| {
        decide(&attempts, &marker, max_retries, &mode, event)
    })
}

fn decide(
    attempts: &AtomicUsize,
    marker: &str,
    max_retries: usize,
    mode: &RetryMode,
    event: HookEvent,
) -> HookDecision {
    let HookEvent::ModelTurnFinished { turn, content, .. } = event else {
        return HookDecision::Continue;
    };

    let should_retry = content.iter().any(
        |content| matches!(content, AssistantContent::Text(text) if text.text.contains(marker)),
    );
    if !should_retry {
        return HookDecision::ModelTurn(ModelTurnAction::continue_run());
    }

    let attempt = attempts.fetch_add(1, Ordering::Relaxed) + 1;
    if attempt > max_retries {
        return HookDecision::ModelTurn(ModelTurnAction::stop(format!(
            "response retry limit ({max_retries}) exceeded"
        )));
    }

    println!("[turn {turn}] rejected response; retry {attempt}/{max_retries}");
    match mode {
        RetryMode::Repeat => HookDecision::ModelTurn(ModelTurnAction::repeat()),
        RetryMode::Feedback(feedback) => {
            HookDecision::ModelTurn(ModelTurnAction::retry_with_feedback(feedback.clone()))
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = openai::Client::from_env()?;
    let agent = client
        .agent(openai::GPT_4O_MINI)
        .preamble(
            "Follow this protocol exactly. For the initial request, reply exactly \
             `RETRY: incomplete draft`. If the latest user message asks you to \
             replace the rejected response, reply exactly `ACCEPTED`.",
        )
        .build();

    // These could just as easily come from a config file or database; the hook
    // owns them and does not require leaked or string-literal references.
    let retry_marker = ["RETRY", ":"].concat();
    let feedback = format!(
        "Replace the rejected response. Reply exactly `{}`.",
        "ACCEPTED"
    );
    let response = agent
        .runner("Begin the retry-hook demonstration.")
        .max_turns(2)
        .add_hook(retry_on_marker(
            retry_marker,
            1,
            RetryMode::Feedback(feedback),
        ))
        .run()
        .await?;

    println!("Final response: {}", response.output);
    println!("Model calls: {}", response.completion_calls.len());

    // Repeat is a distinct policy: it discards the rejected response and reuses
    // the prompt and preceding history, while freshly preparing the next
    // request. It is configured here but not run because this deterministic
    // protocol deliberately returns the same marker each time.
    let _repeat_agent = client
        .agent(openai::GPT_4O_MINI)
        .default_max_turns(2)
        .add_hook(retry_on_marker("RETRY:", 1, RetryMode::Repeat))
        .build();

    Ok(())
}
