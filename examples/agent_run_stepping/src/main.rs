//! Two complementary ways to drive the agent loop.
//!
//! ## Part 1 — hand-driving a configured agent with [`Agent::drive`]
//!
//! `agent.prompt(...)` runs the sans-IO [`AgentRun`] machine internally;
//! driving it yourself lets you inspect every model call, own the provider
//! transport, execute tools with your own policy, and — because the run state
//! is fully serializable between steps — pause a run while tool calls are
//! pending and resume it later, even in another process. The driver owns the
//! run/turn pairing (request preparation, tool snapshots, structured-output
//! bookkeeping); every side effect stays in this loop.
//!
//! ## Part 2 — high-level [`rig::agent::AgentRunner`] with hooks
//!
//! For the common case you don't need that level of control: attach an
//! [`AgentHook`] to observe tool calls (and every other event) without
//! hand-driving the loop. Use `agent.runner(prompt).add_hook(h).run().await`.
//!
//! Both approaches are demonstrated in `main` below.
//!
//! Requires `OPENAI_API_KEY`.

use anyhow::Result;
use rig::agent::run::{AgentRun, ModelTurnOutcome};
use rig::agent::{
    AgentHook, DriveStep, HookContext, InvalidToolCallAction, ToolCall as ToolCallEvent,
    ToolCallAction,
};
use rig::prelude::*;
use rig::providers::openai;
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize)]
struct OperationArgs {
    x: i32,
    y: i32,
}

#[derive(Debug, thiserror::Error)]
#[error("math error")]
struct MathError;

struct Add;

impl Tool for Add {
    const NAME: &'static str = "add";
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    fn description(&self) -> String {
        "Add x and y together".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "x": { "type": "number", "description": "The first number to add" },
                "y": { "type": "number", "description": "The second number to add" }
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
// A minimal AgentHook that logs every tool call routed through the runner.
// Used in Part 2 below to show the high-level hook-based path.
// ---------------------------------------------------------------------------

struct ToolLoggerHook;

impl AgentHook for ToolLoggerHook {
    async fn on_tool_call(&self, _ctx: &HookContext, event: ToolCallEvent<'_>) -> ToolCallAction {
        println!("[hook] tool call: {}({})", event.tool_name, event.args);
        ToolCallAction::run()
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let openai = openai::Client::from_env()?;
    let model = openai.completion_model(openai::GPT_4O);
    let agent = rig::agent::AgentBuilder::new(model)
        .preamble("You are a calculator. Always use the provided tools to compute results.")
        .default_max_turns(2)
        .tool(Add)
        .build();

    // The driver seeds the run from the agent's configuration (turn budget,
    // tool choice, output schema) and owns the run/turn pairing. Every side
    // effect — the provider call, tool execution — stays in this loop, and no
    // agent hooks run.
    let mut driver = agent.drive("What is 2 + 5?");

    loop {
        match driver.next_step().await? {
            DriveStep::SendRequest { request, turn, .. } => {
                println!("→ model call #{turn}");
                let response = request.send().await?;
                let mut outcome = driver.model_response(&response)?;
                while let ModelTurnOutcome::NeedsResolution(context) = outcome {
                    eprintln!("model called unknown tool `{}`", context.tool_name);
                    // Preserve the agent loop's default fail-fast behavior; a
                    // driver could instead retry, repair, or skip here.
                    outcome = driver.resolve_invalid_tool_call(InvalidToolCallAction::fail())?;
                }
            }
            DriveStep::ExecuteTools { .. } => {
                // The run state is serializable while tool calls are pending:
                // persist it here to pause for approval and resume later —
                // even in a process that never saw this step. Resuming
                // rebuilds the driver from the same agent; the resumed driver
                // re-emits the pending calls and re-derives its dispatch
                // snapshot (tool implementations are live objects, so a fresh
                // process dispatches against its own registry).
                let suspended = serde_json::to_string(driver.run())?;
                let resumed: AgentRun = serde_json::from_str(&suspended)?;
                driver = agent.drive_run(resumed);
                let DriveStep::ExecuteTools { calls, tools } = driver.next_step().await? else {
                    anyhow::bail!("a resumed run re-emits its pending tool calls");
                };

                let mut context = rig::tool::ToolContext::new();
                let mut results = Vec::new();
                for call in &calls {
                    println!(
                        "→ executing {}({})",
                        call.tool_call.function.name, call.tool_call.function.arguments
                    );
                    // `execute_call` honors pre-resolved results (from invalid
                    // tool-call recovery) and dispatches through the exact
                    // snapshot the provider saw advertised for this turn.
                    results.push(tools.execute_call(call, &mut context).await);
                }
                driver.tool_results(results)?;
            }
            DriveStep::Done(response) => {
                println!("✓ {}", response.output);
                println!(
                    "  {} model call(s), {} total tokens",
                    response.completion_calls.len(),
                    response.usage.total_tokens
                );
                break;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Part 2 — high-level AgentRunner path with hooks
    //
    // Most use-cases don't need the manual stepping above. `agent.runner(…)`
    // returns an `AgentRunner` that drives the same machine internally while
    // firing an `AgentHook` at every observable point. Attach hooks with
    // `.add_hook(h)`; each call appends another hook to the stack.
    // -----------------------------------------------------------------------

    println!("\n--- Part 2: AgentRunner with ToolLoggerHook ---");

    let resp = agent
        .runner("What is 2 + 5?")
        .max_turns(2)
        .add_hook(ToolLoggerHook)
        .run()
        .await?;

    println!("✓ {}", resp.output);
    println!(
        "  {} model call(s), {} total tokens",
        resp.completion_calls.len(),
        resp.usage.total_tokens
    );

    Ok(())
}
