//! # Forcing a tool on the first turn: a `RequestPatch` footgun and its fix
//!
//! A hook can steer a single model turn by returning
//! [`CompletionCallAction::patch`] from its completion-call method. A common wish
//! is "make the model call
//! a tool *first*", done by patching `tool_choice = Required`.
//!
//! **The footgun.** A [`RequestPatch`] is **per-turn and non-sticky**: the
//! `CompletionCall` event re-fires on *every* turn, so a hook that patches
//! `Required` unconditionally forces a tool call on *every* turn. The model never
//! reaches a turn where it is free to stop calling tools and write the final
//! answer, so the run loops until `max_turns` and fails with
//! [`PromptError::MaxTurnsError`].
//!
//! **The fix.** Gate the patch on the turn index — force `Required` only on the
//! first turn (`ctx.turn() == 1`). The model is nudged to call the tool up front;
//! later turns inherit the agent's baseline (`auto`), so it can stop and answer.
//!
//! This example runs the footgun first (and catches the resulting
//! `MaxTurnsError`), then runs the fix.
//!
//! Requires `OPENAI_API_KEY`.

use anyhow::Result;
use rig::agent::{AgentHook, CompletionCallAction, HookContext, RequestPatch};
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{CompletionModel, Prompt, PromptError};
use rig::message::ToolChoice;
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

#[derive(Clone)]
struct Add;

impl Tool for Add {
    const NAME: &'static str = "add";
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
    ) -> Result<Self::Output, rig::tool::ToolExecutionError> {
        Ok(args.x + args.y)
    }
}

// ---------------------------------------------------------------------------
// The footgun: force `Required` on EVERY completion call.
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct ForceToolEveryTurn;

impl<M> AgentHook<M> for ForceToolEveryTurn
where
    M: CompletionModel,
{
    async fn on_completion_call(
        &self,
        _ctx: &HookContext,
        _event: rig::agent::hook::CompletionCall<'_>,
    ) -> CompletionCallAction {
        CompletionCallAction::patch(RequestPatch::new().tool_choice(ToolChoice::Required))
    }
}

// ---------------------------------------------------------------------------
// The fix: force `Required` on the FIRST turn only.
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct ForceToolOnFirstTurn;

impl<M> AgentHook<M> for ForceToolOnFirstTurn
where
    M: CompletionModel,
{
    async fn on_completion_call(
        &self,
        ctx: &HookContext,
        _event: rig::agent::hook::CompletionCall<'_>,
    ) -> CompletionCallAction {
        if ctx.turn() == 1 {
            CompletionCallAction::patch(RequestPatch::new().tool_choice(ToolChoice::Required))
        } else {
            CompletionCallAction::cont()
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = openai::Client::from_env()?;
    // A fresh agent per run (both share the same tool and preamble).
    let make_agent = || {
        client
            .agent(openai::GPT_4O)
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
        .prompt(PROMPT)
        .max_turns(4)
        .add_hook(ForceToolOnFirstTurn)
        .await?;
    println!("final answer: {answer}");

    Ok(())
}
