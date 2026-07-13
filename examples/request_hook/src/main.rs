//! Demonstrates the composing hook model with `AgentHook`. Several hooks are
//! stacked via `.add_hook(…).add_hook(…)` and, crucially, **all of them run** —
//! a request patch from one hook no longer short-circuits the others:
//!
//! - `LoggingHook` — observe-only. Registered first so a later terminate could
//!   never hide events from it. Reads the run-scoped [`HookContext`] (run id,
//!   turn, streaming flag).
//! - `ContextHook` — injects an extra context document for the turn via
//!   `RequestPatch::extra_context` (passive RAG).
//! - `SamplingHook` — lowers the sampling temperature for the turn via
//!   `RequestPatch::temperature`.
//! - `TurnCounterHook` — counts completion calls using the run-scoped
//!   `HookContext` scratchpad instead of its own interior mutability.
//!
//! On each `CompletionCall`, the patches from `ContextHook` and `SamplingHook`
//! are **merged in registration order** into one effective patch (see the
//! per-field merge rules on `RequestPatch`), so both take effect on the same
//! turn — and `TurnCounterHook` still runs afterwards. Only an event-specific
//! terminal action (for example `CompletionCallAction::Stop`) stops the stack.
//!
//! Requires `OPENAI_API_KEY`.

use anyhow::Result;
use rig::agent::{AgentHook, CompletionCallAction, HookContext, ObserveAction, RequestPatch};
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{CompletionModel, CompletionResponse, Document, Message, Prompt};
use rig::message::UserContent;
use rig::providers::openai;

// ---------------------------------------------------------------------------
// Hook 1: LoggingHook — observe-only. Reads run-scoped identity from the context.
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct LoggingHook;

impl<M> AgentHook<M> for LoggingHook
where
    M: CompletionModel,
{
    async fn on_completion_call(
        &self,
        ctx: &HookContext,
        event: rig::agent::hook::CompletionCall<'_>,
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
        CompletionCallAction::cont()
    }

    async fn on_completion_response(
        &self,
        ctx: &HookContext,
        event: rig::agent::HookCompletionResponse<'_, M>,
    ) -> ObserveAction {
        let response: &CompletionResponse<M::Response> = event.response;
        println!(
            "[run {}] received response: {:?}",
            ctx.run_id(),
            response.choice
        );
        ObserveAction::cont()
    }
}

// ---------------------------------------------------------------------------
// Hook 2: ContextHook — injects an extra context document for the turn.
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct ContextHook;

impl<M> AgentHook<M> for ContextHook
where
    M: CompletionModel,
{
    async fn on_completion_call(
        &self,
        _ctx: &HookContext,
        _event: rig::agent::hook::CompletionCall<'_>,
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

impl<M> AgentHook<M> for SamplingHook
where
    M: CompletionModel,
{
    async fn on_completion_call(
        &self,
        _ctx: &HookContext,
        _event: rig::agent::hook::CompletionCall<'_>,
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

impl<M> AgentHook<M> for TurnCounterHook
where
    M: CompletionModel,
{
    async fn on_completion_call(
        &self,
        ctx: &HookContext,
        _event: rig::agent::hook::CompletionCall<'_>,
    ) -> CompletionCallAction {
        let n = ctx.scratchpad().update(|c: &mut TurnCount| {
            c.0 += 1;
            c.0
        });
        println!("[turn-counter] completion call #{n} this run");
        CompletionCallAction::cont()
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    let agent = openai::Client::from_env()?
        .agent(openai::GPT_4O)
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
        .await?;

    println!("\nFinal response:\n{response}");

    Ok(())
}
