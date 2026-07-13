//! Classifying tool failures as structured facts and applying policy in hooks.
//!
//! `SystemProbe` simulates Erik Tews's two failure cases: disk I/O (`EIO`) and
//! network unreachable (`ENETUNREACH`). The tool classifies each error and adds
//! typed operation metadata; it does not decide whether the agent may continue.
//!
//! A narrow completion-call hook reliably invokes `system_probe` on turn 1,
//! while the prompt requests the desired operation. It does nothing on later
//! turns, leaving the recoverable run free to produce a final answer. Two tool-result
//! hooks then run in registration order:
//!
//! 1. `FailureRecorder` copies the event's call ID, structured error, and typed
//!    result metadata into a run-scoped scratchpad ledger.
//! 2. `FatalFailurePolicy` looks up that same call ID, terminating on `Other`/`EIO`
//!    while allowing `Network`/`ENETUNREACH` feedback to reach the model. Correlation
//!    matters because results from concurrent tool calls can interleave.
//!
//! `ToolResultEvent` carries facts about one execution: `raw_result` contains the
//! standard classification and `tool_context` holds tool/application-specific typed
//! metadata that is never sent to the model. The scratchpad is different: it is
//! shared, run-scoped hook state. Here it lets one hook record facts for the next
//! hook without coupling either hook to model-visible result text.
//!
//! Live commands (require `OPENAI_API_KEY`):
//!
//! ```text
//! cargo run -p tool_result_outcomes -- fatal
//! cargo run -p tool_result_outcomes -- recoverable
//! ```
//!
//! The fatal command terminates after the disk failure. The recoverable command
//! lets the network failure return to the model. `--help` requires no credentials.

use anyhow::{Result, bail};
use rig::agent::{
    AgentHook, CompletionCallAction, CompletionCallEvent, HookContext, RequestPatch,
    ToolResultAction, ToolResultEvent,
};
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{CompletionModel, Prompt};
use rig::message::ToolChoice;
use rig::providers::openai;
use rig::tool::{Tool, ToolContext, ToolErrorKind, ToolExecutionError, ToolResult};

#[derive(Clone, Copy, Debug, serde::Deserialize, serde::Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Operation {
    ReadDisk,
    ConnectNetwork,
}

impl Operation {
    const fn as_str(self) -> &'static str {
        match self {
            Self::ReadDisk => "read_disk",
            Self::ConnectNetwork => "connect_network",
        }
    }
}

#[derive(serde::Deserialize)]
struct ProbeArgs {
    operation: Operation,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FailureSite {
    operation: Operation,
    resource: &'static str,
}

#[derive(Debug, thiserror::Error)]
enum ProbeError {
    #[error("disk read failed for /data/archive.bin")]
    DiskIo,
    #[error("network is unreachable for backup.example.net")]
    NetworkUnreachable,
}

struct SystemProbe;

impl Tool for SystemProbe {
    const NAME: &'static str = "system_probe";
    type Args = ProbeArgs;
    type Output = String;

    fn description(&self) -> String {
        "Run a simulated system operation. Use read_disk for disk access or connect_network for remote access."
            .to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["read_disk", "connect_network"]
                }
            },
            "required": ["operation"]
        })
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, ToolExecutionError> {
        let (error, site) = match args.operation {
            Operation::ReadDisk => (
                ProbeError::DiskIo,
                FailureSite {
                    operation: args.operation,
                    resource: "/data/archive.bin",
                },
            ),
            Operation::ConnectNetwork => (
                ProbeError::NetworkUnreachable,
                FailureSite {
                    operation: args.operation,
                    resource: "backup.example.net",
                },
            ),
        };
        context.insert_result(site);

        let message = error.to_string();
        let execution_error = match error {
            ProbeError::DiskIo => ToolExecutionError::other(message)
                .with_model_feedback("the requested disk operation failed")
                .with_code("EIO")
                .with_retryable(false)
                .with_source(ProbeError::DiskIo),
            ProbeError::NetworkUnreachable => ToolExecutionError::network(message)
                .with_model_feedback("the backup service is unreachable; try again later")
                .with_code("ENETUNREACH")
                .with_source(ProbeError::NetworkUnreachable),
        };
        Err(execution_error)
    }
}

struct ForceSystemProbeOnFirstTurn;

fn system_probe_patch(turn: usize) -> Option<RequestPatch> {
    (turn == 1).then(|| {
        RequestPatch::new().tool_choice(ToolChoice::Specific {
            function_names: vec![SystemProbe::NAME.to_string()],
        })
    })
}

impl<M> AgentHook<M> for ForceSystemProbeOnFirstTurn
where
    M: CompletionModel,
{
    async fn on_completion_call(
        &self,
        _ctx: &HookContext,
        event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        match system_probe_patch(event.turn) {
            Some(patch) => CompletionCallAction::patch(patch),
            None => CompletionCallAction::continue_run(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FailureRecord {
    internal_call_id: String,
    tool_name: String,
    kind: ToolErrorKind,
    code: Option<String>,
    operation: Operation,
    resource: &'static str,
}

#[derive(Clone, Default)]
struct FailureLedger(Vec<FailureRecord>);

struct FailureRecorder;

fn failure_record(
    internal_call_id: &str,
    tool_name: &str,
    result: &ToolResult,
    tool_context: &ToolContext,
) -> Option<FailureRecord> {
    let (Some(error), Some(site)) = (result.error(), tool_context.result::<FailureSite>()) else {
        return None;
    };

    Some(FailureRecord {
        internal_call_id: internal_call_id.to_string(),
        tool_name: tool_name.to_string(),
        kind: error.kind(),
        code: error.code().map(str::to_string),
        operation: site.operation,
        resource: site.resource,
    })
}

impl<M> AgentHook<M> for FailureRecorder
where
    M: CompletionModel,
{
    async fn on_tool_result(
        &self,
        ctx: &HookContext,
        event: ToolResultEvent<'_>,
    ) -> ToolResultAction {
        let Some(record) = failure_record(
            event.internal_call_id,
            event.tool_name,
            event.raw_result,
            event.tool_context,
        ) else {
            return ToolResultAction::keep();
        };
        println!(
            "[recorder] {} {} failed: kind={}, code={}, resource={}",
            record.tool_name,
            record.operation.as_str(),
            record.kind,
            record.code.as_deref().unwrap_or("none"),
            record.resource
        );
        ctx.scratchpad()
            .update(|ledger: &mut FailureLedger| ledger.0.push(record));
        ToolResultAction::keep()
    }
}

#[derive(Debug, PartialEq, Eq)]
enum PolicyDecision {
    Fatal(String),
    Recoverable,
}

fn decide(record: &FailureRecord) -> PolicyDecision {
    if record.kind == ToolErrorKind::Other && record.code.as_deref() == Some("EIO") {
        PolicyDecision::Fatal(format!(
            "fatal disk I/O failure from {} ({})",
            record.tool_name, record.resource
        ))
    } else {
        PolicyDecision::Recoverable
    }
}

fn policy_action(ledger: Option<&FailureLedger>, internal_call_id: &str) -> ToolResultAction {
    let decision = ledger
        .and_then(|ledger| {
            ledger
                .0
                .iter()
                .rev()
                .find(|record| record.internal_call_id == internal_call_id)
        })
        .map(decide);

    match decision {
        Some(PolicyDecision::Fatal(reason)) => ToolResultAction::stop(reason),
        Some(PolicyDecision::Recoverable) => {
            println!("[policy] recoverable failure; returning feedback to the model");
            ToolResultAction::keep()
        }
        None => ToolResultAction::keep(),
    }
}

struct FatalFailurePolicy;

impl<M> AgentHook<M> for FatalFailurePolicy
where
    M: CompletionModel,
{
    async fn on_tool_result(
        &self,
        ctx: &HookContext,
        event: ToolResultEvent<'_>,
    ) -> ToolResultAction {
        if event.raw_result.error().is_none() {
            return ToolResultAction::keep();
        }

        let ledger = ctx.scratchpad().get::<FailureLedger>();
        policy_action(ledger.as_ref(), event.internal_call_id)
    }
}

#[derive(Clone, Copy)]
enum Mode {
    Fatal,
    Recoverable,
}

fn usage() {
    println!(
        "Usage: tool_result_outcomes <fatal|recoverable>\n\n\
         fatal       simulate disk EIO; policy terminates the run\n\
         recoverable simulate ENETUNREACH; model receives tool feedback"
    );
}

fn parse_mode() -> Result<Option<Mode>> {
    match std::env::args().nth(1).as_deref() {
        Some("fatal") => Ok(Some(Mode::Fatal)),
        Some("recoverable") => Ok(Some(Mode::Recoverable)),
        Some("-h" | "--help") | None => Ok(None),
        Some(other) => bail!("unknown mode `{other}`; use --help"),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let Some(mode) = parse_mode()? else {
        usage();
        return Ok(());
    };

    let (operation, prompt) = match mode {
        Mode::Fatal => (
            "read_disk",
            "Use system_probe with read_disk exactly once, then report the result.",
        ),
        Mode::Recoverable => (
            "connect_network",
            "Use system_probe with connect_network exactly once, then explain the failure without retrying.",
        ),
    };
    println!("Running simulated {operation} path");

    let agent = openai::Client::from_env()?
        .agent(openai::GPT_4O)
        .preamble("Follow the user's requested system_probe operation exactly.")
        .tool(SystemProbe)
        .build();

    let response = agent
        .prompt(prompt)
        .max_turns(2)
        .add_hook(ForceSystemProbeOnFirstTurn)
        .add_hook(FailureRecorder)
        .add_hook(FatalFailurePolicy)
        .await?;
    println!("\nFinal response:\n{response}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rig::tool::ToolSet;

    async fn structured_failure(operation: Operation) -> (ToolResult, ToolContext) {
        let tools = ToolSet::from_tools(vec![SystemProbe]);
        let mut context = ToolContext::new();
        let args = serde_json::json!({ "operation": operation }).to_string();
        let result = tools.execute(SystemProbe::NAME, args, &mut context).await;
        (result, context)
    }

    #[test]
    fn system_probe_is_forced_on_first_turn_only() {
        assert!(system_probe_patch(0).is_none());
        let first_turn = system_probe_patch(1);
        assert_eq!(
            first_turn.and_then(|patch| patch.tool_choice),
            Some(ToolChoice::Specific {
                function_names: vec![SystemProbe::NAME.to_string()],
            })
        );
        assert!(system_probe_patch(2).is_none());
        assert!(system_probe_patch(3).is_none());
    }

    #[tokio::test]
    async fn connect_network_preserves_classification_and_typed_metadata() {
        let (result, context) = structured_failure(Operation::ConnectNetwork).await;
        assert!(result.error().is_some(), "probe should fail");
        let Some(error) = result.error() else {
            return;
        };
        assert_eq!(error.kind(), ToolErrorKind::Network);
        assert_eq!(error.code(), Some("ENETUNREACH"));
        assert_eq!(result.output().as_text(), error.model_feedback());
        assert_eq!(
            context.result::<FailureSite>(),
            Some(&FailureSite {
                operation: Operation::ConnectNetwork,
                resource: "backup.example.net",
            })
        );
    }

    #[tokio::test]
    async fn recorder_data_drives_fatal_and_recoverable_actions_by_call_id() {
        let (fatal_result, fatal_context) = structured_failure(Operation::ReadDisk).await;
        let (recoverable_result, recoverable_context) =
            structured_failure(Operation::ConnectNetwork).await;

        let mut ledger = FailureLedger::default();
        let fatal_record = failure_record(
            "fatal-call",
            SystemProbe::NAME,
            &fatal_result,
            &fatal_context,
        );
        let recoverable_record = failure_record(
            "recoverable-call",
            SystemProbe::NAME,
            &recoverable_result,
            &recoverable_context,
        );
        assert!(fatal_record.is_some(), "fatal failure record");
        assert!(recoverable_record.is_some(), "recoverable failure record");
        let (Some(fatal_record), Some(recoverable_record)) = (fatal_record, recoverable_record)
        else {
            return;
        };
        ledger.0.push(fatal_record);
        // Interleave a later recoverable record: policy must not use `last()`.
        ledger.0.push(recoverable_record);

        assert!(matches!(
            policy_action(Some(&ledger), "fatal-call"),
            ToolResultAction::Stop(_)
        ));
        assert_eq!(
            policy_action(Some(&ledger), "recoverable-call"),
            ToolResultAction::Keep
        );
        assert_eq!(
            policy_action(Some(&ledger), "missing-call"),
            ToolResultAction::Keep
        );
        assert_eq!(policy_action(None, "fatal-call"), ToolResultAction::Keep);
    }

    #[tokio::test]
    async fn missing_metadata_cannot_create_a_record_or_reuse_stale_state() {
        let (result, _context) = structured_failure(Operation::ReadDisk).await;
        assert!(
            failure_record(
                "current-call",
                SystemProbe::NAME,
                &result,
                &ToolContext::new(),
            )
            .is_none()
        );

        let stale = FailureLedger(vec![FailureRecord {
            internal_call_id: "stale-call".to_string(),
            tool_name: SystemProbe::NAME.to_string(),
            kind: ToolErrorKind::Other,
            code: Some("EIO".to_string()),
            operation: Operation::ReadDisk,
            resource: "/stale",
        }]);
        assert_eq!(
            policy_action(Some(&stale), "current-call"),
            ToolResultAction::Keep
        );
    }
}
