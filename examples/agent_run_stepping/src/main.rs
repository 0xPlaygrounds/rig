//! Two complementary ways to drive the agent loop.
//!
//! ## Part 1 — hand-driven [`AgentRun`] state machine
//!
//! `agent.prompt(...)` runs this machine internally; stepping it yourself lets
//! you inspect every model call, execute tools with your own policy, and —
//! because the machine is fully serializable between steps — pause a run while
//! tool calls are pending and resume it later (even in another process).
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

use std::collections::BTreeSet;

use anyhow::Result;
use rig::agent::run::{AgentRun, AgentRunStep, ModelTurn, ModelTurnOutcome};
use rig::agent::{
    AgentHook, HookContext, InvalidToolCallAction, ToolCall as ToolCallEvent, ToolCallAction,
};
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::CompletionModel;
use rig::message::UserContent;
use rig::providers::openai;
use rig::tool::{Tool, ToolSet};
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

impl<M: CompletionModel> AgentHook<M> for ToolLoggerHook {
    async fn on_tool_call(&self, _ctx: &HookContext, event: ToolCallEvent<'_>) -> ToolCallAction {
        println!("[hook] tool call: {}({})", event.tool_name, event.args);
        ToolCallAction::run()
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let openai = openai::Client::from_env()?;
    let model = openai.completion_model(openai::GPT_4O);
    let agent = rig::agent::AgentBuilder::new(model.clone())
        .preamble("You are a calculator. Always use the provided tools to compute results.")
        .tool(Add)
        .build();
    let local_tools = ToolSet::builder().static_tool(Add).build();
    let tool_definitions = local_tools.get_tool_definitions();

    let mut run = AgentRun::new("What is 2 + 5?").max_turns(2);

    loop {
        match run.next_step()? {
            AgentRunStep::CallModel {
                prompt,
                history,
                turn,
            } => {
                println!("→ model call #{turn}");
                // A hand-driven `AgentRun` is a sans-IO protocol primitive, not
                // execution of the configured `Agent`. Its transport is an
                // explicit raw model request and therefore has no agent hooks.
                let response = model
                    .completion_request(prompt)
                    .messages(history)
                    .preamble(
                        "You are a calculator. Always use the provided tools to compute results."
                            .to_string(),
                    )
                    .tools(tool_definitions.clone())
                    .send()
                    .await?;

                // The tools advertised to the provider for this turn. With
                // static tools these are the agent's registered tools; agents
                // with dynamic (RAG) tools would resolve them per turn.
                let tool_names: BTreeSet<String> = tool_definitions
                    .iter()
                    .map(|def| def.name.clone())
                    .collect();

                let mut outcome = run.model_response(ModelTurn::new(
                    response.message_id.clone(),
                    response.choice.clone(),
                    response.usage,
                    tool_names.clone(),
                    tool_names,
                ))?;
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
                // re-emits the pending tool calls from its own state.
                let suspended = serde_json::to_string(&run)?;
                let mut run_resumed: AgentRun = serde_json::from_str(&suspended)?;
                let AgentRunStep::CallTools { calls } = run_resumed.next_step()? else {
                    anyhow::bail!("resumed run must re-emit the pending tool calls");
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
                    let result = local_tools.execute(name, args, &mut context).await;
                    results.push(UserContent::tool_result(
                        call.tool_call.id.clone(),
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
