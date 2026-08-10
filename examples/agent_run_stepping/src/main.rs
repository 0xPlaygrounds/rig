//! Two complementary ways to drive the agent loop.
//!
//! ## Part 1 — hand-driven [`AgentRun`] state machine
//!
//! `agent.prompt(...)` runs this machine internally; stepping it yourself lets
//! you inspect every model call, execute tools with your own policy, and —
//! because the machine is fully serializable between steps — pause a run while
//! tool calls are pending and resume it later (even in another process).
//!
//! The per-turn request comes from [`Agent::prepare_turn`]: the loop reuses the
//! configured `Agent`'s preamble, tools, and model parameters instead of
//! restating them, and dispatches tool calls through the same registry snapshot
//! the provider saw advertised.
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
use rig::agent::TurnTools;
use rig::agent::run::{AgentRun, AgentRunStep, ModelTurn, ModelTurnOutcome};
use rig::agent::{
    AgentHook, HookContext, InvalidToolCallAction, ToolCall as ToolCallEvent, ToolCallAction,
};
use rig::message::UserContent;
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
        .tool(Add)
        .build();

    let mut run = AgentRun::new("What is 2 + 5?").max_turns(2);
    // The tool sets and dispatch target of the most recent prepared turn. Tool
    // calls always execute through the snapshot whose definitions the provider
    // saw — the same guarantee the runner gives its own turns.
    let mut turn_tools: Option<TurnTools> = None;

    loop {
        match run.next_step()? {
            AgentRunStep::CallModel {
                prompt,
                history,
                turn,
            } => {
                println!("→ model call #{turn}");
                // A hand-driven `AgentRun` is a sans-IO protocol primitive, not
                // execution of the configured `Agent`: the driver owns the IO
                // and no agent hooks run. `prepare_turn` supplies the request —
                // preamble, tools, model parameters — from the agent's
                // configuration instead of restating it here.
                let (request, tools) = agent.prepare_turn(prompt, &history).await?.into_parts();
                let response = request.send().await?;

                let mut outcome = run.model_response(ModelTurn::new(
                    response.message_id.clone(),
                    response.choice.clone(),
                    response.usage,
                    tools.executable_tool_names().clone(),
                    tools.allowed_tool_names().clone(),
                ))?;
                turn_tools = Some(tools);
                while let ModelTurnOutcome::NeedsResolution(context) = outcome {
                    eprintln!("model called unknown tool `{}`", context.tool_name);
                    // Preserve the agent loop's default fail-fast behavior; a
                    // driver could instead retry, repair, or skip here.
                    outcome = run.resolve_invalid_tool_call(InvalidToolCallAction::fail())?;
                }
            }
            AgentRunStep::CallTools { .. } => {
                // The whole run is serializable while tool calls are pending:
                // persist it here to pause for approval and resume later —
                // even in a process that never saw this step. The resumed run
                // re-emits the pending tool calls from its own state. (Tool
                // implementations are live objects: a genuinely separate
                // process would rebuild the same `Agent` and prepare its own
                // turns from there.)
                let suspended = serde_json::to_string(&run)?;
                let mut run_resumed: AgentRun = serde_json::from_str(&suspended)?;
                let AgentRunStep::CallTools { calls } = run_resumed.next_step()? else {
                    anyhow::bail!("resumed run must re-emit the pending tool calls");
                };
                let Some(tools) = turn_tools.as_ref() else {
                    anyhow::bail!("CallTools always follows a prepared CallModel turn");
                };

                let mut results = Vec::new();
                for call in calls {
                    // Tool calls suppressed by invalid tool-call recovery come
                    // with a pre-resolved result and must not be executed.
                    if let Some(result) = call.preresolved_result {
                        results.push(result);
                        continue;
                    }
                    let name = &call.tool_call.function.name;
                    let args = call.tool_call.function.arguments.to_string();
                    println!("→ executing {name}({args})");
                    let mut context = rig::tool::ToolContext::new();
                    let result = tools.execute(name, &args, &mut context).await;
                    results.push(UserContent::tool_result_for(
                        call.tool_call.id.clone(),
                        call.tool_call.provider.clone(),
                        name.clone(),
                        result.output().clone().into_content(),
                    ));
                }
                run_resumed.tool_results(results)?;
                run = run_resumed;
            }
            AgentRunStep::Done(response) => {
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
