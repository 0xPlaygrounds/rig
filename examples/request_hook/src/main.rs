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
//! turn — and `TurnCounterHook` still runs afterwards. Only a terminal action
//! (`Flow::Terminate`, or a fail-closed misuse) stops the stack.
//!
//! Requires `OPENAI_API_KEY`.

use anyhow::Result;
use rig::agent::{AgentHook, Flow, HookContext, RequestPatch, StepEvent};
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
    async fn on_event(&self, ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
        match event {
            StepEvent::CompletionCall { prompt, .. } => {
                if let Message::User { content } = prompt {
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
                Flow::cont()
            }
            StepEvent::CompletionResponse { response, .. } => {
                let response: &CompletionResponse<M::Response> = response;
                println!(
                    "[run {}] received response: {:?}",
                    ctx.run_id(),
                    response.choice
                );
                Flow::cont()
            }
            _ => Flow::cont(),
        }
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
    async fn on_event(&self, _ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
        if matches!(event, StepEvent::CompletionCall { .. }) {
            let doc = Document {
                id: "style-guide".to_string(),
                text: "House style: keep jokes short and family-friendly.".to_string(),
                additional_props: Default::default(),
            };
            // Appended after the agent's static/dynamic context, for this turn only.
            return Flow::patch_request(RequestPatch::new().context(doc));
        }
        Flow::cont()
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
    async fn on_event(&self, _ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
        if matches!(event, StepEvent::CompletionCall { .. }) {
            return Flow::patch_request(RequestPatch::new().temperature(0.2));
        }
        Flow::cont()
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
    async fn on_event(&self, ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
        if let StepEvent::CompletionCall { .. } = event {
            // Run-scoped state without rolling our own Arc<Mutex<…>>.
            let n = ctx.scratchpad().update(|c: &mut TurnCount| {
                c.0 += 1;
                c.0
            });
            println!("[turn-counter] completion call #{n} this run");
        }
        Flow::cont()
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
