//! # Forcing a tool on the first turn: a `RequestPatch` footgun and its fix
//!
//! A hook is an attach-and-forget record — a [`HookEntry`] wrapping a closure
//! over owned [`HookEvent`] values — and it can steer a single model turn by
//! returning [`CompletionCallAction::patch`] for the
//! [`HookEvent::BeforeModelCall`] event. A common wish is "make the model call
//! a tool *first*", done by patching `tool_choice = Required`.
//!
//! **The footgun.** A [`RequestPatch`] is **per-turn and non-sticky**:
//! `BeforeModelCall` re-fires on *every* turn, so a hook that patches
//! `Required` unconditionally forces a tool call on *every* turn. The model never
//! reaches a turn where it is free to stop calling tools and write the final
//! answer, so the run loops until `max_turns` and fails with
//! [`PromptError::MaxTurnsError`].
//!
//! **The fix.** Gate the patch on the event's own `turn` field — force
//! `Required` only on the first turn. The model is nudged to call the tool up
//! front; later turns inherit the agent's baseline (`auto`), so it can stop and
//! answer.
//!
//! (Only the turn-scoped events are delivered by default: an entry that wants to
//! watch streaming `TextDelta` / `ToolCallDelta` events must opt in with
//! `HookEntry::observing_deltas`.)
//!
//! This example runs the footgun first (and catches the resulting
//! `MaxTurnsError`), then runs the fix.
//!
//! Requires `OPENAI_API_KEY`.

use anyhow::Result;
use rig::agent::{CompletionCallAction, RequestPatch};
use rig::completion::PromptError;
use rig::hooks::{HookDecision, HookEntry, HookEvent};
use rig::message::ToolChoice;
use rig::prelude::*;
use rig::providers::openai;
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::json;

const PREAMBLE: &str =
    "You are a calculator assistant. Use the add tool for arithmetic, then report the result.";
const PROMPT: &str = "What is 21 + 21? Use the add tool, then tell me the answer.";

// ---------------------------------------------------------------------------
// A tiny calculator tool the hook can force the model to call.
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct AddArgs {
    x: i64,
    y: i64,
}

#[derive(Debug, thiserror::Error)]
#[error("math error")]
struct MathError;

#[derive(Clone)]
struct Add;

impl Tool for Add {
    const NAME: &'static str = "add";
    type Error = MathError;
    type Args = AddArgs;
    type Output = i64;

    fn description(&self) -> String {
        "Add x and y together".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "x": { "type": "number", "description": "The first addend" },
                "y": { "type": "number", "description": "The second addend" }
            },
            "required": ["x", "y"]
        })
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(args.x + args.y)
    }
}

// ---------------------------------------------------------------------------
// The footgun: force `Required` on EVERY completion call.
// ---------------------------------------------------------------------------

fn force_tool_every_turn() -> HookEntry {
    HookEntry::new("force-tool-every-turn", |event| {
        let decision = match event {
            HookEvent::BeforeModelCall { .. } => HookDecision::CompletionCall(
                CompletionCallAction::patch(RequestPatch::new().tool_choice(ToolChoice::Required)),
            ),
            // Every other event is none of this entry's business.
            _ => HookDecision::Continue,
        };
        Box::pin(async move { decision })
    })
}

// ---------------------------------------------------------------------------
// The fix: force `Required` on the FIRST turn only, by matching the event's
// own `turn` field.
// ---------------------------------------------------------------------------

fn force_tool_on_first_turn() -> HookEntry {
    HookEntry::new("force-tool-on-first-turn", |event| {
        let decision = match event {
            HookEvent::BeforeModelCall { turn: 1, .. } => HookDecision::CompletionCall(
                CompletionCallAction::patch(RequestPatch::new().tool_choice(ToolChoice::Required)),
            ),
            HookEvent::BeforeModelCall { .. } => {
                HookDecision::CompletionCall(CompletionCallAction::continue_run())
            }
            _ => HookDecision::Continue,
        };
        Box::pin(async move { decision })
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    let cfg = openai::functions::Config::from_env(openai::GPT_4O)?;
    // A fresh agent per run (both share the same tool and preamble). The
    // provider config is plain data, so each build just clones it.
    let make_agent = || {
        AgentBuilder::new(cfg.clone())
            .preamble(PREAMBLE)
            .tool(Add)
            .build()
    };

    // 1) The footgun. Forcing `Required` on every turn re-forces a tool call each
    //    turn, so the run loops until `max_turns` and errors.
    println!("=== forcing tool_choice=Required on EVERY turn (the footgun) ===");
    let agent = make_agent();
    match agent
        .runner(PROMPT)
        .max_turns(4)
        .add_hook(force_tool_every_turn())
        .run()
        .await
        .map(|response| response.output)
    {
        Ok(answer) => println!("(unexpected) got a final answer: {answer}\n"),
        Err(PromptError::MaxTurnsError { max_turns, .. }) => println!(
            "hit MaxTurnsError after {max_turns} model calls — every turn re-forced a tool call, so \
             the model never produced a final answer.\n"
        ),
        Err(err) => println!("run failed: {err}\n"),
    }

    // 2) The fix. Forcing `Required` on the first turn only nudges the model to
    //    call the tool up front, then lets it answer.
    println!("=== forcing tool_choice=Required on the FIRST turn only (the fix) ===");
    let agent = make_agent();
    let answer = agent
        .runner(PROMPT)
        .max_turns(4)
        .add_hook(force_tool_on_first_turn())
        .run()
        .await?
        .output;
    println!("final answer: {answer}");

    Ok(())
}
