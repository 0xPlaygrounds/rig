//! Forces a tool call on the first turn only.
//!
//! A [`RequestPatch`] returned with [`CompletionCallAction::patch`] applies to one
//! turn, and the [`CompletionCallEvent`] fires on every turn. Patching
//! `tool_choice = Required` unconditionally forces a tool on every turn, so the run
//! never answers and fails with [`PromptError::MaxTurns`]. Gating the patch on
//! `ctx.turn() == 1` lets later turns fall back to `auto`. The example runs the
//! unconditional hook first, catching `MaxTurns`, then the gated one.
//!
//! Requires `OPENAI_API_KEY`.

use anyhow::Result;
use rig::agent::{AgentHook, CompletionCallAction, CompletionCallEvent, HookContext, RequestPatch};
use rig::completion::PromptError;
use rig::message::ToolChoice;
use rig::prelude::*;
use rig::providers::openai::{self, OpenAI};
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

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Ok(args.x + args.y)
    }
}

// ---------------------------------------------------------------------------
// The footgun: force `Required` on EVERY completion call.
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct ForceToolEveryTurn;

impl AgentHook for ForceToolEveryTurn {
    async fn on_completion_call(
        &self,
        _ctx: &HookContext,
        _event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        CompletionCallAction::patch(RequestPatch::new().tool_choice(ToolChoice::Required))
    }
}

// ---------------------------------------------------------------------------
// The fix: force `Required` on the FIRST turn only.
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct ForceToolOnFirstTurn;

impl AgentHook for ForceToolOnFirstTurn {
    async fn on_completion_call(
        &self,
        ctx: &HookContext,
        _event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        if ctx.turn() == 1 {
            CompletionCallAction::patch(RequestPatch::new().tool_choice(ToolChoice::Required))
        } else {
            CompletionCallAction::continue_run()
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = OpenAI::from_env()?;
    // A fresh agent per run (both share the same tool and preamble).
    let make_agent = || {
        AgentBuilder::new(client.completion(openai::GPT_4O))
            .preamble(PREAMBLE)
            .tool(Add)
            .build()
    };

    // 1) The footgun. Forcing `Required` on every turn re-forces a tool call each
    //    turn, so the run loops until `max_turns` and errors.
    println!("=== forcing tool_choice=Required on EVERY turn (the footgun) ===");
    let agent = make_agent();
    match agent
        .prompt(PROMPT)
        .max_turns(4)
        .add_hook(ForceToolEveryTurn)
        .await
    {
        Ok(answer) => println!("(unexpected) got a final answer: {}\n", answer.output()),
        Err(PromptError::MaxTurns { max_turns, .. }) => println!(
            "hit the max-turns limit after {max_turns} model calls — every turn re-forced a tool call, so \
             the model never produced a final answer.\n"
        ),
        Err(err) => println!("run failed: {err}\n"),
    }

    // 2) The fix. Forcing `Required` on the first turn only nudges the model to
    //    call the tool up front, then lets it answer.
    println!("=== forcing tool_choice=Required on the FIRST turn only (the fix) ===");
    let agent = make_agent();
    let answer = agent
        .prompt(PROMPT)
        .max_turns(4)
        .add_hook(ForceToolOnFirstTurn)
        .await?
        .output();
    println!("final answer: {answer}");

    Ok(())
}
