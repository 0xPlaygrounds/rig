//! Classifying tool failures as structured facts and applying policy in hooks.
//!
//! `SystemProbe` simulates Erik Tews's two failure cases: disk I/O (`EIO`) and
//! network unreachable (`ENETUNREACH`). The tool classifies each error and
//! carries typed failure metadata on the error itself; it does not decide
//! whether the agent may continue.
//!
//! Hooks are attach-and-forget records — a named `HookEntry` wrapping a closure
//! over owned `HookEvent` values — so policy lives in plain functions, not in a
//! trait impl. A narrow completion-call hook reliably invokes `system_probe` on
//! turn 1, while the prompt requests the desired operation. It does nothing on
//! later turns, leaving the recoverable run free to produce a final answer. Two
//! tool-result hooks then run in registration order:
//!
//! 1. `failure_recorder` copies the event's call ID, structured error, and typed
//!    failure metadata into a shared ledger.
//! 2. `fatal_failure_policy` looks up that same call ID, terminating on `Other`/`EIO`
//!    while allowing `Network`/`ENETUNREACH` feedback to reach the model. Correlation
//!    matters because results from concurrent tool calls can interleave.
//!
//! `HookEvent::ToolResult` carries facts about one execution: `result` contains
//! the standard classification (kind, code, model feedback), and the concrete
//! tool error attached via `with_source` can be recovered with
//! `ToolExecutionError::downcast_ref` for tool-specific typed metadata that is
//! never sent to the model. The ledger is different: it is state the *host*
//! owns and both closures capture (an `Arc<Mutex<_>>`): run-scoped hook state
//! now lives with the host, not inside the hook system. Here it lets one hook
//! record facts for the next hook without coupling either hook to
//! model-visible result text.
//!
//! (Neither hook observes streaming deltas; an entry that wants
//! `HookEvent::TextDelta` / `ToolCallDelta` must be built with
//! `.observing_deltas()` or it never receives them.)
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
use rig::agent::{CompletionCallAction, RequestPatch, ToolResultAction};
use rig::hooks::{HookDecision, HookEntry, HookEvent};
use rig::message::ToolChoice;
use rig::prelude::*;
use rig::providers::openai;
use rig::tool::{Tool, ToolErrorKind, ToolExecutionError, ToolResult};
use std::sync::{Arc, Mutex};

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

impl ProbeError {
    /// Typed, model-invisible metadata describing where the failure happened.
    const fn site(&self) -> FailureSite {
        match self {
            Self::DiskIo => FailureSite {
                operation: Operation::ReadDisk,
                resource: "/data/archive.bin",
            },
            Self::NetworkUnreachable => FailureSite {
                operation: Operation::ConnectNetwork,
                resource: "backup.example.net",
            },
        }
    }
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

    fn map_error(&self, error: Self::Error) -> ToolExecutionError {
        match error {
            error @ ProbeError::DiskIo => ToolExecutionError::other(error.to_string())
                .with_model_feedback("the requested disk operation failed")
                .with_code("EIO")
                .with_retryable(false)
                .with_source(error),
            error @ ProbeError::NetworkUnreachable => {
                ToolExecutionError::network(error.to_string())
                    .with_model_feedback("the backup service is unreachable; try again later")
                    .with_code("ENETUNREACH")
                    .with_source(error)
            }
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Err(match args.operation {
            Operation::ReadDisk => ProbeError::DiskIo,
            Operation::ConnectNetwork => ProbeError::NetworkUnreachable,
        })
    }
}

fn system_probe_patch(turn: usize) -> Option<RequestPatch> {
    (turn == 1).then(|| {
        RequestPatch::new().tool_choice(ToolChoice::Specific {
            function_names: vec![SystemProbe::NAME.to_string()],
        })
    })
}

/// Forces `system_probe` on turn 1 only; every other event is left alone.
fn force_system_probe_on_first_turn() -> HookEntry {
    HookEntry::new("force-system-probe", |event| {
        let decision = match event {
            HookEvent::BeforeModelCall { turn, .. } => {
                HookDecision::CompletionCall(match system_probe_patch(turn) {
                    Some(patch) => CompletionCallAction::patch(patch),
                    None => CompletionCallAction::continue_run(),
                })
            }
            _ => HookDecision::Continue,
        };
        Box::pin(async move { decision })
    })
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

#[derive(Clone, Debug, Default)]
struct FailureLedger(Vec<FailureRecord>);

/// The ledger the two hooks share. The host owns it and both closures capture a
/// clone, which is how run-scoped hook state works now that hooks are plain
/// records rather than trait objects.
#[derive(Clone, Default)]
struct SharedLedger(Arc<Mutex<FailureLedger>>);

fn failure_record(
    internal_call_id: &str,
    tool_name: &str,
    result: &ToolResult,
) -> Option<FailureRecord> {
    let error = result.error()?;
    // Tool-specific typed metadata travels on the error source, never to the
    // model. Downcast recovers it without parsing model-visible text.
    let site = error.downcast_ref::<ProbeError>().map(ProbeError::site)?;

    Some(FailureRecord {
        internal_call_id: internal_call_id.to_string(),
        tool_name: tool_name.to_string(),
        kind: error.kind(),
        code: error.code().map(str::to_string),
        operation: site.operation,
        resource: site.resource,
    })
}

/// Records structured failures into the shared ledger; never steers the run.
fn failure_recorder(ledger: SharedLedger) -> HookEntry {
    HookEntry::new("failure-recorder", move |event| {
        let decision = match event {
            HookEvent::ToolResult {
                call,
                internal_call_id,
                result,
                ..
            } => {
                if let Some(record) =
                    failure_record(&internal_call_id, &call.function.name, &result)
                {
                    println!(
                        "[recorder] {} {} failed: kind={}, code={}, resource={}",
                        record.tool_name,
                        record.operation.as_str(),
                        record.kind,
                        record.code.as_deref().unwrap_or("none"),
                        record.resource
                    );
                    if let Ok(mut ledger) = ledger.0.lock() {
                        ledger.0.push(record);
                    }
                }
                HookDecision::ToolResult(ToolResultAction::keep())
            }
            _ => HookDecision::Continue,
        };
        Box::pin(async move { decision })
    })
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

/// Reads the ledger the recorder wrote, correlating by `internal_call_id`, and
/// terminates the run on a fatal classification.
fn fatal_failure_policy(ledger: SharedLedger) -> HookEntry {
    HookEntry::new("fatal-failure-policy", move |event| {
        let decision = match event {
            HookEvent::ToolResult {
                internal_call_id,
                result,
                ..
            } => {
                let action = if result.error().is_none() {
                    ToolResultAction::keep()
                } else {
                    match ledger.0.lock() {
                        Ok(ledger) => policy_action(Some(&*ledger), &internal_call_id),
                        // A poisoned ledger means the recorder panicked; fall
                        // back to the no-record path (keep the result).
                        Err(_) => policy_action(None, &internal_call_id),
                    }
                };
                HookDecision::ToolResult(action)
            }
            _ => HookDecision::Continue,
        };
        Box::pin(async move { decision })
    })
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

    let cfg = openai::functions::Config::from_env(openai::GPT_4O)?;
    let agent = AgentBuilder::new(ProviderConfig::OpenAi(cfg))
        .preamble("Follow the user's requested system_probe operation exactly.")
        .tool(SystemProbe)
        .build();

    // The recorder must see the result before the policy reads its ledger, so
    // the entries are registered in that order.
    let ledger = SharedLedger::default();
    let response = agent
        .runner(prompt)
        .max_turns(2)
        .add_hook(force_system_probe_on_first_turn())
        .add_hook(failure_recorder(ledger.clone()))
        .add_hook(fatal_failure_policy(ledger))
        .run()
        .await?
        .output;
    println!("\nFinal response:\n{response}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rig::tool::PortableDynamicTool;

    async fn structured_failure(operation: Operation) -> ToolResult {
        let tool = PortableDynamicTool::from_portable(SystemProbe);
        let args = serde_json::json!({ "operation": operation });
        match tool.execute(args).await {
            Ok(output) => ToolResult::success(output),
            Err(error) => ToolResult::failed(error),
        }
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
        let result = structured_failure(Operation::ConnectNetwork).await;
        assert!(result.error().is_some(), "probe should fail");
        let Some(error) = result.error() else {
            return;
        };
        assert_eq!(error.kind(), ToolErrorKind::Network);
        assert_eq!(error.code(), Some("ENETUNREACH"));
        assert_eq!(result.output().as_text(), error.model_feedback());
        assert_eq!(
            error.downcast_ref::<ProbeError>().map(ProbeError::site),
            Some(FailureSite {
                operation: Operation::ConnectNetwork,
                resource: "backup.example.net",
            })
        );
    }

    #[tokio::test]
    async fn recorder_data_drives_fatal_and_recoverable_actions_by_call_id() {
        let fatal_result = structured_failure(Operation::ReadDisk).await;
        let recoverable_result = structured_failure(Operation::ConnectNetwork).await;

        let mut ledger = FailureLedger::default();
        let fatal_record = failure_record("fatal-call", SystemProbe::NAME, &fatal_result);
        let recoverable_record =
            failure_record("recoverable-call", SystemProbe::NAME, &recoverable_result);
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
        // An error without the tool's typed source carries no failure site, so
        // no record can be created from it.
        let untyped =
            ToolResult::failed(ToolExecutionError::other("disk read failed").with_code("EIO"));
        assert!(failure_record("current-call", SystemProbe::NAME, &untyped).is_none());

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
