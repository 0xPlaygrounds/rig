//! Three complementary ways to drive the agent loop.
//!
//! ## Part 1 — hand-driven [`AgentRun`] fed by a configured [`rig::agent::Agent`]
//!
//! `agent.prompt(...)` runs this machine internally; stepping it yourself lets
//! you inspect every model call and execute tools with your own policy. The
//! agent stays the single source of configuration and tools:
//! `agent.new_run(...)` seeds the run with the agent's durable policy, and
//! `agent.prepare_completion_request(...)` builds each hook-free provider
//! request straight from the `CallModel` step. The returned
//! [`rig::agent::PreparedAgentTurn`] pairs the response and the turn's tool
//! calls with the exact tool-name sets and implementation snapshot the request
//! advertised.
//!
//! The relationship:
//!
//! ```text
//! Agent
//!   | new_run(prompt)
//!   +-------------------------------> AgentRun            (durable policy)
//!   |
//!   | prepare_completion_request(prompt, history, &mut run)
//!   v
//! PreparedAgentRequest
//!   | into_parts()
//!   +----> request                    caller sends through its transport
//!   `----> PreparedAgentTurn          (in-process, one issued turn)
//!           | model_turn(response)    exact name-set metadata
//!           ` execute_call(...)       exact pinned implementation snapshot
//!
//! After cross-process resume only:
//! rebuilt Agent::tool_server_handle() live dispatch, live-registry semantics
//! ```
//!
//! This surface runs no hooks, memory, telemetry, or concurrency policy: the
//! manual driver owns provider IO, retry bounds, result correlation,
//! persistence timing, and any approval policy.
//!
//! ## Part 2 — cross-process resume
//!
//! The run state is fully serializable while tool calls are pending. A
//! restarted process rebuilds the agent, deserializes the run, and finishes
//! the pending calls through the rebuilt agent's live
//! `tool_server_handle()` — the prepared turn is in-process state for one
//! issued request and deliberately does not survive.
//!
//! ## Part 3 — high-level [`rig::agent::AgentRunner`] with hooks
//!
//! For the common case you don't need that level of control: attach an
//! [`AgentHook`] to observe tool calls (and every other event) without
//! hand-driving the loop. Use `agent.runner(prompt).add_hook(h).run().await`.
//!
//! Requires `OPENAI_API_KEY`.

use anyhow::Result;
use rig::agent::run::{AgentRun, AgentRunStep, ModelTurnOutcome};
use rig::agent::{
    Agent, AgentHook, HookContext, InvalidToolCallAction, ToolCall as ToolCallEvent, ToolCallAction,
};
use rig::completion::CompletionModel;
use rig::prelude::*;
use rig::providers::openai;
use rig::tool::{Tool, ToolContext};
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
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Ok(args.x + args.y)
    }
}

/// Build the calculator agent. Part 2 calls this twice to simulate a process
/// restart: the "restarted process" reconstructs an equivalent agent.
fn build_agent(model: impl CompletionModel + 'static) -> Agent {
    rig::agent::AgentBuilder::new(model)
        .preamble("You are a calculator. Always use the provided tools to compute results.")
        .tool(Add)
        .build()
}

// ---------------------------------------------------------------------------
// A minimal AgentHook that logs every tool call routed through the runner.
// Used in Part 3 below to show the high-level hook-based path.
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
    let agent = build_agent(model.clone());

    // -----------------------------------------------------------------------
    // Part 1 — manual loop over the agent-seeded run.
    // -----------------------------------------------------------------------

    // Seed the run with the agent's durable policy (tool choice, output
    // validation, default turn budget); `AgentRun`'s builder methods still
    // apply on top.
    let mut run = agent.new_run("What is 2 + 5?").max_turns(2);
    // The prepared turn from the latest model call, retained across the send
    // so the following `CallTools` step dispatches through the same pinned
    // implementation snapshot whose definitions the provider saw.
    let mut prepared_turn = None;
    // One tool context for the whole run, like the classic runner: result
    // metadata from each dispatch is published back into it.
    let mut context = ToolContext::new();

    loop {
        // Both `AgentRunStep` and `ModelTurnOutcome` are deliberately
        // exhaustive: a driver must handle every variant.
        match run.next_step()? {
            AgentRunStep::CallModel {
                prompt,
                history,
                turn,
            } => {
                println!("→ model call #{turn}");
                // Prepare a hook-free request directly from the step's fields.
                // The agent supplies preamble, context, sampling parameters,
                // tool definitions, and tool choice; impossible tool choices
                // fail here, before any provider IO.
                let (request, turn_pairing) = agent
                    .prepare_completion_request(prompt, history, &mut run)
                    .await?
                    .into_parts();
                // The caller owns the send (its transport, retries, and
                // errors). `model_turn` then supplies the exact executable and
                // allowed tool-name sets the request advertised — nothing is
                // re-derived from the live registry.
                let response = request.send().await?;
                let mut outcome = run.model_response(turn_pairing.model_turn(response))?;
                prepared_turn = Some(turn_pairing);
                loop {
                    match outcome {
                        ModelTurnOutcome::Continue { .. } => break,
                        ModelTurnOutcome::TurnRetried => break,
                        ModelTurnOutcome::NeedsResolution(context) => {
                            eprintln!("model called unknown tool `{}`", context.tool_name);
                            // Preserve the agent loop's default fail-fast
                            // behavior; a driver could instead retry, repair,
                            // or skip here.
                            outcome =
                                run.resolve_invalid_tool_call(InvalidToolCallAction::fail())?;
                        }
                    }
                }
            }
            AgentRunStep::CallTools { calls } => {
                let Some(turn_pairing) = prepared_turn.as_ref() else {
                    anyhow::bail!("CallTools always follows a prepared model call");
                };
                let mut results = Vec::with_capacity(calls.len());
                for call in &calls {
                    println!(
                        "→ executing {}({})",
                        call.tool_call.function.name, call.tool_call.function.arguments
                    );
                    // `execute_call` honors pre-resolved results (from invalid
                    // tool-call recovery) and dispatches everything else
                    // through the snapshot pinned at preparation — a tool
                    // re-registered since then cannot be reached, and a tool
                    // registered since then is rejected.
                    results.push(turn_pairing.execute_call(call, &mut context).await);
                }
                run.tool_results(results)?;
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
    // Part 2 — cross-process resume.
    //
    // Drive a fresh run up to its first pending tool calls, serialize it, and
    // "restart": rebuild the agent and finish through its live handle.
    // -----------------------------------------------------------------------

    println!("\n--- Part 2: suspend with pending tool calls, resume after a restart ---");

    let mut run = agent.new_run("What is 3 + 4?").max_turns(2);
    let suspended = loop {
        match run.next_step()? {
            AgentRunStep::CallModel {
                prompt, history, ..
            } => {
                let (request, turn_pairing) = agent
                    .prepare_completion_request(prompt, history, &mut run)
                    .await?
                    .into_parts();
                let response = request.send().await?;
                match run.model_response(turn_pairing.model_turn(response))? {
                    ModelTurnOutcome::Continue { .. } | ModelTurnOutcome::TurnRetried => {}
                    ModelTurnOutcome::NeedsResolution(context) => {
                        anyhow::bail!("unexpected unknown tool `{}`", context.tool_name);
                    }
                }
            }
            // Suspend while tool calls are pending: the serialized run is
            // self-contained and re-emits them on resume.
            AgentRunStep::CallTools { .. } => break serde_json::to_string(&run)?,
            AgentRunStep::Done(response) => {
                anyhow::bail!(
                    "expected a tool call before completion, got: {}",
                    response.output
                )
            }
        }
    };
    drop(run);
    drop(agent);

    // "Restarted process": rebuild an equivalent agent and deserialize the
    // run. The `PreparedAgentTurn` did not survive — it pins an in-process
    // implementation snapshot for one issued request and is deliberately not
    // serializable — so pending calls dispatch through the rebuilt agent's
    // live tool_server_handle(), under live-registry semantics.
    let agent = build_agent(model);
    let mut run: AgentRun = serde_json::from_str(&suspended)?;
    let handle = agent.tool_server_handle();
    // As in Part 1: one tool context for the whole (resumed) run.
    let mut context = ToolContext::new();

    loop {
        match run.next_step()? {
            AgentRunStep::CallTools { calls } => {
                let mut results = Vec::with_capacity(calls.len());
                for call in &calls {
                    // Tool calls suppressed by invalid tool-call recovery come
                    // with a pre-resolved result and must not be executed.
                    if let Some(result) = &call.preresolved_result {
                        results.push(result.clone());
                        continue;
                    }
                    let name = &call.tool_call.function.name;
                    let args = call.tool_call.function.arguments.to_string();
                    println!("→ executing {name}({args}) via the live handle");
                    let result = handle.execute(name, &args, &mut context).await;
                    // `result_content` correlates the executed tool's output
                    // with this call (id, provider call id, tool name), so
                    // none of those fields are copied by hand.
                    results.push(call.result_content(result.output().clone()));
                }
                run.tool_results(results)?;
            }
            AgentRunStep::CallModel {
                prompt, history, ..
            } => {
                let (request, turn_pairing) = agent
                    .prepare_completion_request(prompt, history, &mut run)
                    .await?
                    .into_parts();
                let response = request.send().await?;
                match run.model_response(turn_pairing.model_turn(response))? {
                    ModelTurnOutcome::Continue { .. } | ModelTurnOutcome::TurnRetried => {}
                    ModelTurnOutcome::NeedsResolution(context) => {
                        anyhow::bail!("unexpected unknown tool `{}`", context.tool_name);
                    }
                }
            }
            AgentRunStep::Done(response) => {
                println!("✓ resumed run finished: {}", response.output);
                break;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Part 3 — high-level AgentRunner path with hooks
    //
    // Most use-cases don't need the manual stepping above. `agent.runner(…)`
    // returns an `AgentRunner` that drives the same machine internally while
    // firing an `AgentHook` at every observable point. Attach hooks with
    // `.add_hook(h)`; each call appends another hook to the stack.
    // -----------------------------------------------------------------------

    println!("\n--- Part 3: AgentRunner with ToolLoggerHook ---");

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
