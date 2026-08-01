//! Demonstrates the composing hook model. A hook is now a plain
//! **attach-and-forget record** — a [`HookEntry`] pairing a name with a closure
//! over owned [`HookEvent`] values — and several of them are stacked via
//! `.add_hook(…).add_hook(…)`. Crucially, **all of them run**: a request patch
//! from one entry no longer short-circuits the others.
//!
//! - `logging_entry` — observe-only. Registered first so a later terminate could
//!   never hide events from it. Run-scoped identity is no longer handed to the
//!   hook by the runtime: the host mints a [`RunId`] and the closure simply
//!   captures it, and the turn index arrives on the event itself.
//! - `context_entry` — injects an extra context document for the turn via
//!   `RequestPatch::context` (passive RAG).
//! - `sampling_entry` — lowers the sampling temperature for the turn via
//!   `RequestPatch::temperature`.
//! - `turn_counter_entry` — counts completion calls with host-owned state (an
//!   `Arc<AtomicUsize>` captured by the closure), which is what replaced the old
//!   run-scoped scratchpad.
//!
//! On each `BeforeModelCall`, the patches from `context_entry` and
//! `sampling_entry` are **merged in registration order** into one effective
//! patch (see the per-field merge rules on `RequestPatch`), so both take effect
//! on the same turn — and `turn_counter_entry` still runs afterwards. A typed
//! stop action short-circuits the stack.
//!
//! Note that each entry sees only the events it cares about and returns
//! [`HookDecision::Continue`] for the rest. Streaming `TextDelta` /
//! `ToolCallDelta` events are gated: an entry must be built with
//! [`HookEntry::observing_deltas`] to receive them at all. None of the entries
//! below observe deltas, so none opts in.
//!
//! Requires `OPENAI_API_KEY`.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use rig::agent::{CompletionCallAction, ObservationAction, RequestPatch, RunId};
use rig::completion::{Document, Message};
use rig::hooks::{HookDecision, HookEntry, HookEvent};
use rig::message::UserContent;
use rig::prelude::*;
use rig::providers::openai;

// ---------------------------------------------------------------------------
// Entry 1: observe-only. Captures the host-owned run id; reads the turn from
// the event.
// ---------------------------------------------------------------------------

fn logging_entry(run_id: RunId) -> HookEntry {
    HookEntry::sync("logging", move |event| match event {
        HookEvent::BeforeModelCall { turn, prompt, .. } => {
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
                    println!("[run {run_id} · turn {turn}] sending prompt: {prompt_text}");
                }
            }
            HookDecision::CompletionCall(CompletionCallAction::continue_run())
        }
        HookEvent::CompletionResponse { response, .. } => {
            println!(
                "[run {run_id}] received response (usage: {:?}, message_id: {:?}): {:?}",
                response.usage, response.message_id, response.choice
            );
            HookDecision::Observation(ObservationAction::continue_run())
        }
        _ => HookDecision::Continue,
    })
}

// ---------------------------------------------------------------------------
// Entry 2: injects an extra context document for the turn.
// ---------------------------------------------------------------------------

fn context_entry() -> HookEntry {
    HookEntry::sync("context", |event| match event {
        HookEvent::BeforeModelCall { .. } => {
            let doc = Document {
                id: "style-guide".to_string(),
                text: "House style: keep jokes short and family-friendly.".to_string(),
                additional_props: Default::default(),
            };
            HookDecision::CompletionCall(CompletionCallAction::patch(
                RequestPatch::new().context(doc),
            ))
        }
        _ => HookDecision::Continue,
    })
}

// ---------------------------------------------------------------------------
// Entry 3: lowers the temperature for the turn. Its patch MERGES with
// `context_entry`'s rather than replacing it.
// ---------------------------------------------------------------------------

fn sampling_entry() -> HookEntry {
    HookEntry::sync("sampling", |event| match event {
        HookEvent::BeforeModelCall { .. } => HookDecision::CompletionCall(
            CompletionCallAction::patch(RequestPatch::new().temperature(0.2)),
        ),
        _ => HookDecision::Continue,
    })
}

// ---------------------------------------------------------------------------
// Entry 4: counts completion calls using host-owned state captured by the
// closure (the replacement for the old run-scoped scratchpad).
// ---------------------------------------------------------------------------

fn turn_counter_entry(count: Arc<AtomicUsize>) -> HookEntry {
    HookEntry::sync("turn-counter", move |event| match event {
        HookEvent::BeforeModelCall { .. } => {
            let n = count.fetch_add(1, Ordering::SeqCst) + 1;
            println!("[turn-counter] completion call #{n} this run");
            HookDecision::CompletionCall(CompletionCallAction::continue_run())
        }
        _ => HookDecision::Continue,
    })
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    let client = openai::Client::from_env()?;
    let agent = client
        .agent(openai::GPT_4O)
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .build();

    // Run-scoped identity and state are the host's to own now — mint them here
    // and let the closures capture them.
    let run_id = RunId::generate();
    let turn_count = Arc::new(AtomicUsize::new(0));

    // Attach four hook records. They run in registration order on every event;
    // the two request-patch entries (context, sampling) both contribute to the
    // same turn because `BeforeModelCall` patches accumulate and merge — neither
    // short-circuits the other, and the turn counter still runs after them.
    let response = agent
        .runner("Entertain me!")
        .add_hook(logging_entry(run_id))
        .add_hook(context_entry())
        .add_hook(sampling_entry())
        .add_hook(turn_counter_entry(turn_count.clone()))
        .run()
        .await?
        .output;

    println!("\nFinal response:\n{response}");
    println!(
        "(observed {} completion calls)",
        turn_count.load(Ordering::SeqCst)
    );

    Ok(())
}
