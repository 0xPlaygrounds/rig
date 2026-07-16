//! **Durable** human-in-the-loop tool-call approval.
//!
//! Unlike `agent_with_human_in_the_loop` (which `.await`s the human *inline*
//! inside an `AgentHook`, so the approval only lives as long as the running
//! task), this example pauses the run **across a serialization boundary**: when
//! the model wants to run tools, the entire [`AgentRun`] state machine is written
//! to a JSON file and then reconstructed from that file before the decision is
//! taken. Everything between the write and the reload could be a *different
//! process*, an HTTP request handler, or hours/days later — the run carries no
//! live futures, only serializable state.
//!
//! This is rig's analogue of LangGraph's `interrupt()` + checkpointer, the OpenAI
//! Agents SDK's serializable `RunState`, and pydantic-ai's `DeferredToolRequests`
//! — built on the fact that `AgentRun` is `Serialize + Deserialize` and the run
//! loop is driven by hand (see also `agent_run_stepping`). Each per-call decision
//! maps to ordinary tool-result content:
//!
//! - **approve** → execute the tool, return its real output
//! - **deny**    → don't execute; return the reason as the tool result so the model adapts
//! - **edit**    → execute with human-supplied JSON arguments instead
//! - **abort**   → stop the run
//!
//! The gate is **fail-closed**: empty/unknown input denies, and closed stdin
//! (e.g. piped `< /dev/null`) aborts — a destructive tool never runs on ambiguous
//! input. The prompt is a UX affordance, not a security boundary; enforce real
//! authorization inside the tool.
//!
//! Requires `OPENAI_API_KEY`. Run with: `cargo run -p agent_with_durable_approval`

use anyhow::Result;
use rig::agent::InvalidToolCallAction;
use rig::agent::run::{AgentRun, AgentRunStep, ModelTurn, ModelTurnOutcome};
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::CompletionModel;
use rig::message::{ToolResultContent, UserContent};
use rig::providers::openai;
use rig::tool::{Tool, ToolSet};
use serde::Deserialize;
use serde_json::json;
use std::collections::BTreeSet;

// ---------------------------------------------------------------------------
// One read-only tool and one side-effecting tool worth gating.
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
#[error("tool failed: {0}")]
struct ToolError(String);

#[derive(Deserialize)]
struct BalanceArgs {
    account: String,
}

struct GetBalance;

impl Tool for GetBalance {
    const NAME: &'static str = "get_balance";
    type Error = ToolError;
    type Args = BalanceArgs;
    type Output = String;

    fn description(&self) -> String {
        "Get the balance of an account.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": { "account": { "type": "string", "description": "Account id" } },
            "required": ["account"]
        })
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        println!("   💰 [get_balance] -> {}", args.account);
        Ok(format!("account {} balance: $1000", args.account))
    }
}

#[derive(Deserialize)]
struct TransferArgs {
    to: String,
    amount: u64,
}

struct TransferFunds;

impl Tool for TransferFunds {
    const NAME: &'static str = "transfer_funds";
    type Error = ToolError;
    type Args = TransferArgs;
    type Output = String;

    fn description(&self) -> String {
        "Transfer funds to an account.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "to": { "type": "string", "description": "Destination account id" },
                "amount": { "type": "integer", "description": "Amount in whole dollars" }
            },
            "required": ["to", "amount"]
        })
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        // A real implementation would move money here.
        println!("   🏦 [transfer_funds] -> ${} to {}", args.amount, args.to);
        Ok(format!("transferred ${} to {}", args.amount, args.to))
    }
}

// ---------------------------------------------------------------------------
// Fail-closed stdin reader (None on EOF / closed stdin / read error).
// ---------------------------------------------------------------------------

async fn ask(prompt: &str) -> Option<String> {
    use std::io::Write;
    print!("{prompt}");
    let _ = std::io::stdout().flush();
    let line = tokio::task::spawn_blocking(|| {
        let mut line = String::new();
        match std::io::stdin().read_line(&mut line) {
            Ok(0) | Err(_) => None,
            Ok(_) => Some(line),
        }
    })
    .await
    .ok()
    .flatten()?;
    Some(line.trim().to_ascii_lowercase())
}

#[tokio::main]
async fn main() -> Result<()> {
    // A serializable `AgentRun` is a sans-IO protocol primitive. This example
    // intentionally supplies raw model transport and tool dispatch explicitly;
    // configured `Agent` execution instead always goes through `AgentRunner`.
    let model = openai::Client::from_env()?.completion_model(openai::GPT_4O);
    let preamble = "You are a banking assistant. Use the tools to carry out the user's request. \
                    Call one tool at a time.";
    let tools = ToolSet::builder()
        .static_tool(GetBalance)
        .static_tool(TransferFunds)
        .build();
    let tool_definitions = tools.get_tool_definitions();

    let prompt = "Check the balance of account A-1, then transfer $500 to account B-2.";
    println!("User: {prompt}");

    // Where the suspended run is checkpointed between approval rounds.
    let state_path = std::env::temp_dir().join("rig_durable_approval.json");
    let _ = std::fs::remove_file(&state_path);

    let mut run = AgentRun::new(prompt).max_turns(10);

    loop {
        match run.next_step()? {
            AgentRunStep::CallModel {
                prompt,
                history,
                turn,
            } => {
                println!("\n→ model call #{turn}");
                let response = model
                    .completion_request(prompt)
                    .messages(history)
                    .preamble(preamble.to_string())
                    .tools(tool_definitions.clone())
                    .send()
                    .await?;
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
                    outcome = run.resolve_invalid_tool_call(InvalidToolCallAction::fail())?;
                }
            }

            AgentRunStep::CallTools { .. } => {
                // DURABLE PAUSE. Persist the whole run, then reconstruct it from
                // the file before deciding. The write→reload boundary below could
                // be a separate process / request / much later — the resumed run
                // re-emits the pending tool calls purely from serialized state.
                std::fs::write(&state_path, serde_json::to_vec_pretty(&run)?)?;
                println!("\n💾 run suspended to {}", state_path.display());

                // ----- imagine the process exits here and resumes later -----

                let mut resumed: AgentRun = serde_json::from_slice(&std::fs::read(&state_path)?)?;
                let AgentRunStep::CallTools { calls } = resumed.next_step()? else {
                    anyhow::bail!("resumed run must re-emit the pending tool calls");
                };

                let mut results = Vec::new();
                let mut aborted = false;
                for call in calls {
                    // Calls suppressed by invalid-tool-call recovery come pre-resolved.
                    if let Some(result) = call.preresolved_result {
                        results.push(result);
                        continue;
                    }
                    let id = call.tool_call.id.clone();
                    let name = call.tool_call.function.name.clone();
                    let args = call.tool_call.function.arguments.to_string();

                    println!("\n⏸  approval required: {name}({args})");
                    match ask("     [a]pprove / [d]eny / [e]dit args / a[b]ort? ")
                        .await
                        .as_deref()
                    {
                        Some("a") | Some("approve") => {
                            let execution = tools
                                .execute(&name, args, &mut rig::tool::ToolContext::new())
                                .await;
                            results.push(UserContent::tool_result(
                                id,
                                execution.output().clone().into_content(),
                            ));
                        }
                        Some("e") | Some("edit") => {
                            let edited = ask("     replacement JSON args (single line): ").await;
                            match edited
                                .as_deref()
                                .map(serde_json::from_str::<serde_json::Value>)
                            {
                                Some(Ok(value)) => {
                                    let execution = tools
                                        .execute(
                                            &name,
                                            value.to_string(),
                                            &mut rig::tool::ToolContext::new(),
                                        )
                                        .await;
                                    results.push(UserContent::tool_result(
                                        id,
                                        execution.output().clone().into_content(),
                                    ));
                                }
                                _ => {
                                    println!("     ! no valid JSON; denying instead");
                                    results.push(UserContent::tool_result(
                                        id,
                                        rig::OneOrMany::one(ToolResultContent::text(
                                            "denied: the reviewer supplied no valid JSON to edit with",
                                        )),
                                    ));
                                }
                            }
                        }
                        // Abort on explicit request or closed stdin (None): fail-closed.
                        Some("b") | Some("abort") | None => {
                            aborted = true;
                            break;
                        }
                        // Deny / empty / unknown: fail-closed; the reason reaches the model.
                        other => {
                            let reason = if matches!(other, Some("d") | Some("deny")) {
                                ask("     reason (shown to the model): ")
                                    .await
                                    .filter(|r| !r.is_empty())
                                    .unwrap_or_else(|| "denied by the human reviewer".to_string())
                            } else {
                                "denied: no clear approval given".to_string()
                            };
                            results.push(UserContent::tool_result(
                                id,
                                rig::OneOrMany::one(ToolResultContent::text(reason)),
                            ));
                        }
                    }
                }

                if aborted {
                    let _ = std::fs::remove_file(&state_path);
                    println!("\nrun aborted by the reviewer");
                    return Ok(());
                }

                resumed.tool_results(results)?;
                let _ = std::fs::remove_file(&state_path);
                run = resumed;
            }

            AgentRunStep::Done(response) => {
                println!("\n✓ {}", response.output);
                return Ok(());
            }
        }
    }
}
