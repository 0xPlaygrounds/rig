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
//! 1. `FailureRecorder` turns the event's structured error and typed result
//!    metadata into a `FailureRecord` in the call's scratchpad.
//! 2. `FatalFailurePolicy` reads that record back, terminating on `Other`/`EIO`
//!    while allowing `Network`/`ENETUNREACH` feedback to reach the model.
//!
//! The tool-call `OutcomeEvent` carries facts about one execution: `tool_result()`
//! contains the standard classification and `tool_context()` holds
//! tool/application-specific typed metadata that is never sent to the model.
//! `event.scratchpad` belongs to that one call, so results from concurrent tool
//! calls cannot see each other's records, however their events interleave. Run-wide
//! state, such as a failure budget, would go in `ctx.scratchpad()` instead. The
//! tests drive both hooks through a `HookStack` with a detached `HookContext`.
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
    AgentHook, CompletionCallAction, CompletionCallEvent, HookContext, OutcomeAction, OutcomeEvent,
    RequestPatch,
};
use rig::message::ToolChoice;
use rig::prelude::*;
use rig::providers::openai::{self, OpenAI};
use rig::tool::{Tool, ToolContext, ToolErrorKind, ToolExecutionError, ToolResult, tool_name};

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

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize, rig::ContextValue)]
#[context(key = "example.failure_site")]
struct FailureSite {
    operation: Operation,
    resource: String,
}

#[derive(Debug, thiserror::Error)]
enum ProbeError {
    #[error("disk read failed for /data/archive.bin")]
    DiskIo,
    #[error("network is unreachable for backup.example.net")]
    NetworkUnreachable,
    #[error("tool context: {0}")]
    Context(#[from] rig::tool::ToolContextError),
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
            ProbeError::Context(error) => ToolExecutionError::from(error),
        }
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let (error, site) = match args.operation {
            Operation::ReadDisk => (
                ProbeError::DiskIo,
                FailureSite {
                    operation: args.operation,
                    resource: "/data/archive.bin".to_string(),
                },
            ),
            Operation::ConnectNetwork => (
                ProbeError::NetworkUnreachable,
                FailureSite {
                    operation: args.operation,
                    resource: "backup.example.net".to_string(),
                },
            ),
        };
        context.insert_result(site)?;
        Err(error)
    }
}

struct ForceSystemProbeOnFirstTurn;

fn system_probe_patch(turn: usize) -> Option<RequestPatch> {
    (turn == 1).then(|| {
        RequestPatch::new().tool_choice(ToolChoice::Specific {
            function_names: vec![tool_name::<SystemProbe>()],
        })
    })
}

impl AgentHook for ForceSystemProbeOnFirstTurn {
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
    tool_name: String,
    kind: ToolErrorKind,
    code: Option<String>,
    operation: Operation,
    resource: String,
}

struct FailureRecorder;

fn failure_record(
    tool_name: &str,
    result: &ToolResult,
    tool_context: &ToolContext,
) -> Option<FailureRecord> {
    let (Some(error), Ok(Some(site))) = (result.error(), tool_context.result::<FailureSite>())
    else {
        return None;
    };

    Some(FailureRecord {
        tool_name: tool_name.to_string(),
        kind: error.kind(),
        code: error.code().map(str::to_string),
        operation: site.operation,
        resource: site.resource,
    })
}

impl AgentHook for FailureRecorder {
    async fn on_outcome(&self, _ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
        // `on_outcome` fires for every effect family; only tool results matter here.
        let (Some(tool_name), Some(result), Some(tool_context)) =
            (event.tool_name(), event.tool_result(), event.tool_context())
        else {
            return OutcomeAction::proceed();
        };
        let Some(record) = failure_record(tool_name, result, tool_context) else {
            return OutcomeAction::proceed();
        };
        println!(
            "[recorder] {} {} failed: kind={}, code={}, resource={}",
            record.tool_name,
            record.operation.as_str(),
            record.kind,
            record.code.as_deref().unwrap_or("none"),
            record.resource
        );
        event.scratchpad.insert(record);
        OutcomeAction::proceed()
    }
}

fn policy_action(record: Option<&FailureRecord>) -> OutcomeAction {
    match record {
        Some(record)
            if record.kind == ToolErrorKind::Other && record.code.as_deref() == Some("EIO") =>
        {
            OutcomeAction::stop(format!(
                "fatal disk I/O failure from {} ({})",
                record.tool_name, record.resource
            ))
        }
        Some(_) => {
            println!("[policy] recoverable failure; returning feedback to the model");
            OutcomeAction::proceed()
        }
        None => OutcomeAction::proceed(),
    }
}

struct FatalFailurePolicy;

impl AgentHook for FatalFailurePolicy {
    async fn on_outcome(&self, _ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
        policy_action(event.scratchpad.get::<FailureRecord>().as_ref())
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

    let agent = AgentBuilder::new(OpenAI::from_env()?.completion(openai::GPT_4O))
        .preamble("Follow the user's requested system_probe operation exactly.")
        .tool(SystemProbe)
        .build();

    let response = agent
        .prompt(prompt)
        .max_turns(2)
        .add_hook(ForceSystemProbeOnFirstTurn)
        .add_hook(FailureRecorder)
        .add_hook(FatalFailurePolicy)
        .await?
        .output();
    println!("\nFinal response:\n{response}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rig::ErrorKind;
    use rig::agent::{HookStack, Scratchpad};
    use rig::effect::{Answered, EffectId, EffectKind};
    use rig::tool::ToolSet;

    fn stops_run(action: &OutcomeAction) -> bool {
        matches!(action, OutcomeAction::Replace(Err(report)) if report.kind == ErrorKind::Cancelled)
    }

    async fn structured_failure(operation: Operation) -> (ToolResult, ToolContext) {
        let tools = ToolSet::from_tools(vec![SystemProbe]);
        let mut context = ToolContext::new();
        let args = serde_json::json!({ "operation": operation }).to_string();
        let result = tools.execute(SystemProbe::NAME, args, &mut context).await;
        (result, context)
    }

    fn policy_stack() -> HookStack {
        let mut stack = HookStack::with(FailureRecorder);
        stack.push(FatalFailurePolicy);
        stack
    }

    /// The registered hooks' action on one probe call's outcome, with that
    /// call's own scratchpad.
    async fn outcome_action(
        stack: &HookStack,
        ctx: &HookContext,
        id: u64,
        operation: Operation,
        scratchpad: &Scratchpad,
    ) -> OutcomeAction {
        let (result, context) = structured_failure(operation).await;
        let kind = EffectKind::ToolCall {
            name: SystemProbe::NAME.to_string(),
            args: serde_json::json!({ "operation": operation }).to_string(),
        };
        let outcome = Ok(Answered::ToolResult { result });
        let event = OutcomeEvent {
            id: EffectId::from_raw(id),
            kind: &kind,
            outcome: &outcome,
            turn: 1,
            call_id: None,
            context: Some(&context),
            scratchpad,
        };
        stack.on_outcome(ctx, event).await
    }

    #[test]
    fn system_probe_is_forced_on_first_turn_only() {
        assert!(system_probe_patch(0).is_none());
        let first_turn = system_probe_patch(1);
        assert_eq!(
            first_turn.and_then(|patch| patch.tool_choice),
            Some(ToolChoice::Specific {
                function_names: vec![tool_name::<SystemProbe>()],
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
        let site = context.result::<FailureSite>();
        assert!(site.is_ok(), "decodes: {site:?}");
        assert_eq!(
            site.ok().flatten(),
            Some(FailureSite {
                operation: Operation::ConnectNetwork,
                resource: "backup.example.net".to_string(),
            })
        );
    }

    #[tokio::test]
    async fn each_call_is_judged_by_its_own_record() {
        let (stack, ctx) = (policy_stack(), HookContext::new());
        let (fatal, recoverable) = (Scratchpad::default(), Scratchpad::default());

        let fatal_action = outcome_action(&stack, &ctx, 1, Operation::ReadDisk, &fatal).await;
        // A later recoverable call must not change the fatal call's verdict,
        // nor inherit it.
        let recoverable_action =
            outcome_action(&stack, &ctx, 2, Operation::ConnectNetwork, &recoverable).await;

        assert!(stops_run(&fatal_action));
        assert!(matches!(recoverable_action, OutcomeAction::Proceed));
        assert_eq!(
            fatal.get::<FailureRecord>().map(|record| record.operation),
            Some(Operation::ReadDisk)
        );
        assert_eq!(
            recoverable
                .get::<FailureRecord>()
                .map(|record| record.operation),
            Some(Operation::ConnectNetwork)
        );
    }

    #[tokio::test]
    async fn missing_metadata_creates_no_record() {
        let (result, _context) = structured_failure(Operation::ReadDisk).await;
        assert!(failure_record(SystemProbe::NAME, &result, &ToolContext::new()).is_none());
        assert!(matches!(policy_action(None), OutcomeAction::Proceed));
    }
}
