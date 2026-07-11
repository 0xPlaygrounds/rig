//! Classifying tool failures as structured facts and applying policy in hooks.
//!
//! `SystemProbe` simulates Erik Tews's two failure cases: disk I/O (`EIO`) and
//! network unreachable (`ENETUNREACH`). The tool classifies each error and adds
//! typed operation metadata; it does not decide whether the agent may continue.
//!
//! A narrow `CompletionCall` hook reliably invokes `system_probe` on turn 1,
//! while the prompt requests the desired operation. It does nothing on later
//! turns, leaving the recoverable run free to produce a final answer. Two `ToolResult`
//! hooks then run in registration order:
//!
//! 1. `FailureRecorder` copies the event's call ID, structured outcome, and typed
//!    extension into a run-scoped scratchpad ledger.
//! 2. `FatalFailurePolicy` looks up that same call ID, terminating on `Other`/`EIO`
//!    while allowing `Network`/`ENETUNREACH` feedback to reach the model. Correlation
//!    matters because results from concurrent tool calls can interleave.
//!
//! `StepEvent::ToolResult` carries facts about one execution: `outcome` is the
//! standard classification and `extensions` holds tool/application-specific typed
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
use rig::agent::{AgentHook, Flow, HookContext, RequestPatch, StepEvent};
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{CompletionModel, Prompt};
use rig::message::ToolChoice;
use rig::providers::openai;
use rig::tool::{
    Tool, ToolCallExtensions, ToolFailure, ToolFailureKind, ToolOutcome, ToolResultExtensions,
    ToolReturn,
};

#[derive(Clone, Copy, Debug, serde::Deserialize, PartialEq, Eq)]
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
    type Error = ProbeError;
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

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        match args.operation {
            Operation::ReadDisk => Err(ProbeError::DiskIo),
            Operation::ConnectNetwork => Err(ProbeError::NetworkUnreachable),
        }
    }

    fn classify_error(&self, error: &Self::Error) -> ToolFailure {
        match error {
            ProbeError::DiskIo => ToolFailure::other(error.to_string())
                .with_code("EIO")
                .with_retryable(false),
            ProbeError::NetworkUnreachable => {
                ToolFailure::network(error.to_string()).with_code("ENETUNREACH")
            }
        }
    }

    async fn call_structured(
        &self,
        args: Self::Args,
        _extensions: &ToolCallExtensions,
    ) -> Result<ToolReturn<Self::Output>, Self::Error> {
        let site = match args.operation {
            Operation::ReadDisk => FailureSite {
                operation: args.operation,
                resource: "/data/archive.bin",
            },
            Operation::ConnectNetwork => FailureSite {
                operation: args.operation,
                resource: "backup.example.net",
            },
        };

        match self.call(args).await {
            Ok(output) => Ok(ToolReturn::success(output)),
            Err(error) => {
                let failure = self.classify_error(&error);
                Ok(ToolReturn::failed(error.to_string(), failure).with_extension(site))
            }
        }
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
    async fn on_event(&self, ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
        if matches!(event, StepEvent::CompletionCall { .. })
            && let Some(patch) = system_probe_patch(ctx.turn())
        {
            return Flow::patch_request(patch);
        }
        Flow::cont()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FailureRecord {
    internal_call_id: String,
    tool_name: String,
    kind: ToolFailureKind,
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
    outcome: &ToolOutcome,
    extensions: &ToolResultExtensions,
) -> Option<FailureRecord> {
    let (Some(failure), Some(site)) = (outcome.failure(), extensions.get::<FailureSite>()) else {
        return None;
    };

    Some(FailureRecord {
        internal_call_id: internal_call_id.to_string(),
        tool_name: tool_name.to_string(),
        kind: failure.kind,
        code: failure.code.clone(),
        operation: site.operation,
        resource: site.resource,
    })
}

impl<M> AgentHook<M> for FailureRecorder
where
    M: CompletionModel,
{
    async fn on_event(&self, ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
        let StepEvent::ToolResult {
            internal_call_id,
            tool_name,
            outcome,
            extensions,
            ..
        } = event
        else {
            return Flow::cont();
        };
        let Some(record) = failure_record(internal_call_id, tool_name, outcome, extensions) else {
            return Flow::cont();
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
        Flow::cont()
    }
}

#[derive(Debug, PartialEq, Eq)]
enum PolicyDecision {
    Fatal(String),
    Recoverable,
}

fn decide(record: &FailureRecord) -> PolicyDecision {
    if record.kind == ToolFailureKind::Other && record.code.as_deref() == Some("EIO") {
        PolicyDecision::Fatal(format!(
            "fatal disk I/O failure from {} ({})",
            record.tool_name, record.resource
        ))
    } else {
        PolicyDecision::Recoverable
    }
}

fn policy_flow(ledger: Option<&FailureLedger>, internal_call_id: &str) -> Flow {
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
        Some(PolicyDecision::Fatal(reason)) => Flow::terminate(reason),
        Some(PolicyDecision::Recoverable) => {
            println!("[policy] recoverable failure; returning feedback to the model");
            Flow::cont()
        }
        None => Flow::cont(),
    }
}

struct FatalFailurePolicy;

impl<M> AgentHook<M> for FatalFailurePolicy
where
    M: CompletionModel,
{
    async fn on_event(&self, ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
        let StepEvent::ToolResult {
            internal_call_id,
            outcome,
            ..
        } = event
        else {
            return Flow::cont();
        };
        if outcome.failure().is_none() {
            return Flow::cont();
        }

        let ledger = ctx.scratchpad().get::<FailureLedger>();
        policy_flow(ledger.as_ref(), internal_call_id)
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
        // The recoverable path needs exactly two model calls: one tool call,
        // then one final answer. The fatal path terminates after the first.
        .max_turns(2)
        // This steering hook only guarantees the initial probe; the next two
        // hooks remain the example's recorder/policy pipeline.
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

    async fn structured_failure(operation: Operation) -> Result<ToolReturn<String>, ProbeError> {
        SystemProbe
            .call_structured(ProbeArgs { operation }, &ToolCallExtensions::new())
            .await
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
    async fn connect_network_preserves_classification_and_typed_extension() {
        let returned = structured_failure(Operation::ConnectNetwork).await;
        assert!(returned.is_ok());
        let Ok(returned) = returned else {
            return;
        };
        assert!(returned.outcome.failure().is_some());
        let Some(failure) = returned.outcome.failure() else {
            return;
        };
        assert_eq!(failure.kind, ToolFailureKind::Network);
        assert_eq!(failure.code.as_deref(), Some("ENETUNREACH"));
        assert_eq!(
            returned.extensions.get::<FailureSite>(),
            Some(&FailureSite {
                operation: Operation::ConnectNetwork,
                resource: "backup.example.net",
            })
        );
    }

    #[tokio::test]
    async fn recorder_data_drives_fatal_and_recoverable_flows_by_call_id() {
        let fatal = structured_failure(Operation::ReadDisk).await;
        let recoverable = structured_failure(Operation::ConnectNetwork).await;
        assert!(fatal.is_ok());
        assert!(recoverable.is_ok());
        let (Ok(fatal), Ok(recoverable)) = (fatal, recoverable) else {
            return;
        };
        let fatal_outcome: ToolOutcome = fatal.outcome.into();
        let recoverable_outcome: ToolOutcome = recoverable.outcome.into();

        let mut ledger = FailureLedger::default();
        let fatal_record = failure_record(
            "fatal-call",
            SystemProbe::NAME,
            &fatal_outcome,
            &fatal.extensions,
        );
        let recoverable_record = failure_record(
            "recoverable-call",
            SystemProbe::NAME,
            &recoverable_outcome,
            &recoverable.extensions,
        );
        assert!(fatal_record.is_some());
        assert!(recoverable_record.is_some());
        let (Some(fatal_record), Some(recoverable_record)) = (fatal_record, recoverable_record)
        else {
            return;
        };
        ledger.0.push(fatal_record);
        // Interleave a later recoverable record: policy must not use `last()`.
        ledger.0.push(recoverable_record);

        assert!(matches!(
            policy_flow(Some(&ledger), "fatal-call"),
            Flow::Terminate { .. }
        ));
        assert_eq!(
            policy_flow(Some(&ledger), "recoverable-call"),
            Flow::Continue
        );
        assert_eq!(policy_flow(Some(&ledger), "missing-call"), Flow::Continue);
        assert_eq!(policy_flow(None, "fatal-call"), Flow::Continue);
    }

    #[test]
    fn missing_extension_cannot_create_a_record_or_reuse_stale_state() {
        let outcome = ToolOutcome::Error(
            ToolFailure::other("disk failed")
                .with_code("EIO")
                .with_retryable(false),
        );
        assert!(
            failure_record(
                "current-call",
                SystemProbe::NAME,
                &outcome,
                &ToolResultExtensions::new(),
            )
            .is_none()
        );

        let stale = FailureLedger(vec![FailureRecord {
            internal_call_id: "stale-call".to_string(),
            tool_name: SystemProbe::NAME.to_string(),
            kind: ToolFailureKind::Other,
            code: Some("EIO".to_string()),
            operation: Operation::ReadDisk,
            resource: "/stale",
        }]);
        assert_eq!(policy_flow(Some(&stale), "current-call"), Flow::Continue);
    }
}
