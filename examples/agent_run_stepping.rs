//! Drives the agent loop by hand with the sans-IO [`AgentRun`] state machine.
//!
//! `agent.prompt(...)` runs this machine internally; stepping it yourself lets
//! you inspect every model call, execute tools with your own policy, and —
//! because the machine is fully serializable between steps — pause a run while
//! tool calls are pending and resume it later (even in another process).
//!
//! Requires `OPENAI_API_KEY`.

use std::collections::BTreeSet;

use anyhow::Result;
use rig::agent::InvalidToolCallHookAction;
use rig::agent::run::{AgentRun, AgentRunStep, ModelTurn, ModelTurnOutcome};
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{Completion, ToolDefinition};
use rig::message::{ToolResultContent, UserContent};
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

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "add".to_string(),
            description: "Add x and y together".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "x": { "type": "number", "description": "The first number to add" },
                    "y": { "type": "number", "description": "The second number to add" }
                },
                "required": ["x", "y"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(args.x + args.y)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let openai = openai::Client::from_env()?;
    let agent = openai
        .agent(openai::GPT_4O)
        .preamble("You are a calculator. Always use the provided tools to compute results.")
        .tool(Add)
        .build();

    let mut run = AgentRun::new("What is 2 + 5?").max_turns(2);

    loop {
        match run.next_step()? {
            AgentRunStep::CallModel {
                prompt,
                history,
                turn,
            } => {
                println!("→ model call #{turn}");
                let response = agent.completion(prompt, history).await?.send().await?;

                // The tools advertised to the provider for this turn. With
                // static tools these are the agent's registered tools; agents
                // with dynamic (RAG) tools would resolve them per turn.
                let tool_names: BTreeSet<String> = agent
                    .tool_server_handle
                    .get_tool_defs(None)
                    .await?
                    .into_iter()
                    .map(|def| def.name)
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
                    outcome = run.resolve_invalid_tool_call(InvalidToolCallHookAction::fail())?;
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
                    let output = agent.tool_server_handle.call_tool(name, &args).await?;
                    results.push(UserContent::tool_result(
                        call.tool_call.id.clone(),
                        ToolResultContent::from_tool_output(output),
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

    Ok(())
}
