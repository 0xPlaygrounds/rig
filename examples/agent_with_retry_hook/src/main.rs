//! Retries a completed, tool-free model turn from an [`AgentHook`].
//!
//! `RetryOnMarker` owns its policy limit in the run-scoped [`HookContext`]
//! scratchpad. Rig does not add a separate retry counter to the agent: every
//! retry consumes the request's existing `max_turns` model-call budget.
//!
//! [`RetryMode::Feedback`] preserves the rejected assistant response and adds a
//! corrective user message. [`RetryMode::Repeat`] discards the response and
//! reuses the same prompt and preceding history with fresh request preparation.
//! Completion-call hooks, retrieval, and dynamic tool resolution therefore run
//! again. Tool-bearing turns must instead be handled by tool-call hooks.
//!
//! Requires `OPENAI_API_KEY`. Run with:
//! `cargo run -p agent_with_retry_hook`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use rig::agent::{AgentHook, HookContext, ModelTurnAction, ModelTurnFinished};
use rig::message::AssistantContent;
use rig::prelude::*;
use rig::providers::openai;

static NEXT_RETRY_HOOK_ID: AtomicUsize = AtomicUsize::new(1);

#[derive(Clone, Default)]
struct RetryAttempts(HashMap<usize, usize>);

#[derive(Clone, Copy)]
enum RetryMode {
    Repeat,
    Feedback(&'static str),
}

struct RetryOnMarker {
    id: usize,
    marker: &'static str,
    max_retries: usize,
    mode: RetryMode,
}

impl RetryOnMarker {
    fn with_feedback(marker: &'static str, max_retries: usize, feedback: &'static str) -> Self {
        Self {
            id: NEXT_RETRY_HOOK_ID.fetch_add(1, Ordering::Relaxed),
            marker,
            max_retries,
            mode: RetryMode::Feedback(feedback),
        }
    }

    fn repeat(marker: &'static str, max_retries: usize) -> Self {
        Self {
            id: NEXT_RETRY_HOOK_ID.fetch_add(1, Ordering::Relaxed),
            marker,
            max_retries,
            mode: RetryMode::Repeat,
        }
    }
}

impl AgentHook for RetryOnMarker {
    async fn on_model_turn_finished(
        &self,
        ctx: &HookContext,
        event: ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        let should_retry = event.content.iter().any(|content| {
            matches!(content, AssistantContent::Text(text) if text.text.contains(self.marker))
        });
        if !should_retry {
            return ModelTurnAction::continue_run();
        }

        let attempt = ctx.scratchpad().update(|attempts: &mut RetryAttempts| {
            let attempt = attempts.0.entry(self.id).or_default();
            *attempt += 1;
            *attempt
        });
        if attempt > self.max_retries {
            return ModelTurnAction::stop(format!(
                "response retry limit ({}) exceeded",
                self.max_retries
            ));
        }

        println!(
            "[turn {}] rejected response; retry {attempt}/{}",
            event.turn, self.max_retries
        );
        match self.mode {
            RetryMode::Repeat => ModelTurnAction::repeat(),
            RetryMode::Feedback(feedback) => ModelTurnAction::retry_with_feedback(feedback),
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

    let response = agent
        .runner("Begin the retry-hook demonstration.")
        .max_turns(2)
        .add_hook(RetryOnMarker::with_feedback(
            "RETRY:",
            1,
            "Replace the rejected response. Reply exactly `ACCEPTED`.",
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
        .add_hook(RetryOnMarker::repeat("RETRY:", 1))
        .build();

    Ok(())
}
