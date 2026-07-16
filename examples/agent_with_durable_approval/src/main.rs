//! Durable human approval with runner-owned checkpoint and resumption.
//!
//! `run_until_interruption` stops before a tool batch and returns a serializable
//! `AgentRun`. The example writes that checkpoint to disk, reloads it, asks for
//! approval, and resumes through `AgentRunner`; no raw completion request is
//! prepared or sent by application code.

use anyhow::Result;
use rig::OneOrMany;
use rig::agent::{AgentRun, AgentRunnerOutcome};
use rig::client::{CompletionClient, ProviderClient};
use rig::message::{ToolResultContent, UserContent};
use rig::providers::openai;
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::json;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

#[derive(Debug, thiserror::Error)]
#[error("tool failed")]
struct ToolError;

#[derive(Deserialize)]
struct TransferArgs {
    from: String,
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
        "Transfer funds between accounts".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({"type":"object","properties":{"from":{"type":"string"},"to":{"type":"string"},"amount":{"type":"integer"}},"required":["from","to","amount"]})
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Ok(format!(
            "transferred ${} from {} to {}",
            args.amount, args.from, args.to
        ))
    }
}

async fn approve(summary: &str) -> bool {
    let mut stdout = tokio::io::stdout();
    if stdout
        .write_all(format!("Approve {summary}? [y/N] ").as_bytes())
        .await
        .is_err()
        || stdout.flush().await.is_err()
    {
        return false;
    }
    let mut answer = String::new();
    BufReader::new(tokio::io::stdin())
        .read_line(&mut answer)
        .await
        .is_ok_and(|bytes| bytes > 0)
        && matches!(answer.trim(), "y" | "Y" | "yes")
}

#[tokio::main]
async fn main() -> Result<()> {
    let agent = openai::Client::from_env()?
        .agent(openai::GPT_4O)
        .preamble("Use the transfer tool when requested.")
        .tool(TransferFunds)
        .build();
    let prompt = "Transfer $500 from A-1 to B-2";
    let checkpoint_path = std::env::temp_dir().join("rig-agent-run.json");
    let mut checkpoint: Option<AgentRun> = None;

    loop {
        let mut runner = agent.runner(prompt).max_turns(4);
        if let Some(run) = checkpoint.take() {
            runner = runner.resume(run);
        }

        match runner.run_until_interruption().await? {
            AgentRunnerOutcome::Completed(response) => {
                println!("{}", response.output);
                break;
            }
            AgentRunnerOutcome::Interrupted(run) => {
                std::fs::write(&checkpoint_path, serde_json::to_vec_pretty(&run)?)?;
                println!("checkpointed to {}", checkpoint_path.display());

                // This reload may happen in another process or much later.
                let mut resumed: AgentRun =
                    serde_json::from_slice(&std::fs::read(&checkpoint_path)?)?;
                let calls = resumed
                    .pending_tool_calls()
                    .ok_or_else(|| anyhow::anyhow!("checkpoint must contain pending tools"))?
                    .to_vec();
                let summary = calls
                    .iter()
                    .map(|call| {
                        format!(
                            "{}({})",
                            call.tool_call.function.name, call.tool_call.function.arguments
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ");

                if !approve(&summary).await {
                    let denied = calls
                        .into_iter()
                        .map(|call| {
                            let content =
                                OneOrMany::one(ToolResultContent::text("human approval denied"));
                            match call.tool_call.call_id {
                                Some(call_id) => UserContent::tool_result_with_call_id(
                                    call.tool_call.id,
                                    call_id,
                                    content,
                                ),
                                None => UserContent::tool_result(call.tool_call.id, content),
                            }
                        })
                        .collect();
                    resumed.tool_results(denied)?;
                }
                checkpoint = Some(resumed);
            }
            _ => anyhow::bail!("unsupported runner outcome"),
        }
    }
    Ok(())
}
