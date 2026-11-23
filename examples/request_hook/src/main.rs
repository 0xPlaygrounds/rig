//! Stacks several `AgentHook`s on one request, and every hook runs.
//!
//! `LoggingHook` reads the run's [`HookContext`] and is registered first so a later
//! stop cannot hide events from it. `ContextHook` adds a context document with
//! `RequestPatch::extra_context`, `SamplingHook` lowers the temperature with
//! `RequestPatch::temperature`, and `TurnCounterHook` counts completion calls in
//! the `HookContext` scratchpad. The two patches merge in registration order into
//! one patch for the turn. A stop action would end the stack early.
//!
//! Requires `OPENAI_API_KEY`.

use anyhow::Result;
use rig::agent::{
    AgentHook, CompletionCallAction, CompletionCallEvent, HookContext, OutcomeAction, OutcomeEvent,
    RequestPatch,
};
use rig::completion::{Document, Message};
use rig::message::UserContent;
use rig::prelude::*;
use rig::providers::openai::{self, OpenAI};

// ---------------------------------------------------------------------------
// Hook 1: LoggingHook — observe-only. Reads run-scoped identity from the context.
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct LoggingHook;

impl AgentHook for LoggingHook {
    async fn on_completion_call(
        &self,
        ctx: &HookContext,
        event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        if let Message::User { content } = event.prompt {
            let prompt_text = content
                .iter()
                .filter_map(|c| match c {
                    UserContent::Text(text) => Some(text.text.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            if !prompt_text.is_empty() {
                println!(
                    "[run {} · turn {}] sending prompt: {}",
                    ctx.run_id(),
                    ctx.turn(),
                    prompt_text
                );
            }
        }
        CompletionCallAction::continue_run()
    }

    async fn on_outcome(&self, ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
        // `on_outcome` fires for every effect family; only report completions.
        if let Some(response) = event.completion() {
            println!(
                "[run {}] received response (usage: {:?}, response_id: {:?}): {:?}",
                ctx.run_id(),
                response.usage,
                response.response_id(),
                response.choice
            );
        }
        OutcomeAction::proceed()
    }
}

// ---------------------------------------------------------------------------
// Hook 2: ContextHook — injects an extra context document for the turn.
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct ContextHook;

impl AgentHook for ContextHook {
    async fn on_completion_call(
        &self,
        _ctx: &HookContext,
        _event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        let doc = Document {
            id: "style-guide".to_string(),
            text: "House style: keep jokes short and family-friendly.".to_string(),
            additional_props: Default::default(),
        };
        CompletionCallAction::patch(RequestPatch::new().context(doc))
    }
}

// ---------------------------------------------------------------------------
// Hook 3: SamplingHook — lowers the temperature for the turn. Its patch MERGES
// with ContextHook's rather than replacing it.
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct SamplingHook;

impl AgentHook for SamplingHook {
    async fn on_completion_call(
        &self,
        _ctx: &HookContext,
        _event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        CompletionCallAction::patch(RequestPatch::new().temperature(0.2))
    }
}

// ---------------------------------------------------------------------------
// Hook 4: TurnCounterHook — counts completion calls via the shared scratchpad.
// ---------------------------------------------------------------------------

#[derive(Clone, Default)]
struct TurnCount(usize);

#[derive(Clone)]
struct TurnCounterHook;

impl AgentHook for TurnCounterHook {
    async fn on_completion_call(
        &self,
        ctx: &HookContext,
        _event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        let n = ctx.scratchpad().update(|c: &mut TurnCount| {
            c.0 += 1;
            c.0
        });
        println!("[turn-counter] completion call #{n} this run");
        CompletionCallAction::continue_run()
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    let agent = AgentBuilder::new(OpenAI::from_env()?.completion(openai::GPT_4O))
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .build();

    // Attach four hooks. They run in registration order on every event; the two
    // request-patch hooks (ContextHook, SamplingHook) both contribute to the
    // same turn because CompletionCall patches accumulate and merge — neither
    // short-circuits the other, and TurnCounterHook still runs after them.
    let response = agent
        .prompt("Entertain me!")
        .add_hook(LoggingHook)
        .add_hook(ContextHook)
        .add_hook(SamplingHook)
        .add_hook(TurnCounterHook)
        .await?
        .output();

    println!("\nFinal response:\n{response}");

    Ok(())
}
