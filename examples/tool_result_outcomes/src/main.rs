//! Structured tool errors, metadata, and result-policy hooks.
//!
//! Requires `OPENAI_API_KEY`.

use anyhow::Result;
use rig::agent::{AgentHook, HookContext, HookToolResult, ToolResultAction};
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{CompletionModel, Prompt};
use rig::providers::openai;
use rig::tool::{Tool, ToolContext, ToolErrorKind, ToolExecutionError};
use serde::Deserialize;
use serde_json::json;

#[derive(Clone, Debug)]
struct RequestId(String);

#[derive(Deserialize)]
struct ProbeArgs {
    operation: String,
}

struct SystemProbe;

impl Tool for SystemProbe {
    const NAME: &'static str = "system_probe";
    type Args = ProbeArgs;
    type Output = String;

    fn description(&self) -> String {
        "Probe a named system operation".into()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": { "operation": { "type": "string" } },
            "required": ["operation"]
        })
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, ToolExecutionError> {
        context.insert_metadata(RequestId(format!("probe-{}", args.operation)));
        match args.operation.as_str() {
            "status" => Ok("all systems operational".into()),
            "network" => Err(ToolExecutionError::network("network unreachable")
                .with_code("ENETUNREACH")
                .with_model_feedback("The network is temporarily unavailable; try another task.")),
            "restricted" => Err(
                ToolExecutionError::refused("internal policy refused this probe")
                    .with_model_feedback("That probe is not permitted."),
            ),
            other => Err(ToolExecutionError::not_found(format!(
                "unknown operation `{other}`"
            ))),
        }
    }
}

struct ExecutionPolicy;

impl<M: CompletionModel> AgentHook<M> for ExecutionPolicy {
    async fn on_tool_result(
        &self,
        _context: &HookContext,
        event: HookToolResult<'_>,
    ) -> ToolResultAction {
        let status = event.execution.status();
        let request_id = event
            .execution
            .metadata::<RequestId>()
            .map(|id| id.0.as_str())
            .unwrap_or("none");
        println!(
            "tool={} call={} status={} request_id={request_id}",
            event.tool_name,
            event.internal_call_id,
            status.as_str(),
        );

        if status.is_error_kind(ToolErrorKind::Network) {
            ToolResultAction::rewrite("Temporary connectivity failure (redacted by policy).")
        } else {
            ToolResultAction::keep()
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let agent = openai::Client::from_env()?
        .agent(openai::GPT_4O)
        .preamble("Use system_probe to inspect the requested operation, then explain the result.")
        .tool(SystemProbe)
        .add_hook(ExecutionPolicy)
        .default_max_turns(3)
        .build();

    let response = agent.prompt("Check the network operation.").await?;
    println!("{response}");
    Ok(())
}
