//! The open calls of a `CallTools` step and the answers that close them. The
//! run classifies each call once, and builds every result from the answer.

use serde::{Deserialize, Serialize};

use rig_core::message::{CallId, ToolCall, ToolName};
use rig_core::tool::ToolResult;

use super::InvalidToolCallAction;

/// One open tool call of a [`CallTools`](super::AgentRunStep::CallTools) step.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub enum PendingToolCall {
    /// A call whose arguments parsed and that nothing suppressed: execute it.
    Execute(ExecCall),
    /// A call whose arguments are not a JSON object: answer it without running it.
    Malformed(MalformedCall),
}

/// A tool call the driver should execute, answered with [`ExecCall::answer`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecCall {
    pub(super) turn: usize,
    pub(super) index: usize,
    pub(super) tool_call: ToolCall,
}

impl ExecCall {
    /// The provider's call id.
    pub fn id(&self) -> &CallId {
        &self.tool_call.id
    }

    /// The tool to execute.
    pub fn name(&self) -> &ToolName {
        &self.tool_call.function.name
    }

    /// The call as the model emitted it, with any repaired tool name applied.
    pub fn tool_call(&self) -> &ToolCall {
        &self.tool_call
    }

    /// The parsed arguments, a JSON object.
    pub fn arguments(&self) -> serde_json::Value {
        self.tool_call.function.arguments_value()
    }

    /// Answer this call with the result of executing it (or of a hook
    /// skipping it, as [`ToolResult::skipped`]).
    pub fn answer(self, result: ToolResult) -> ToolAnswer {
        ToolAnswer {
            turn: self.turn,
            index: self.index,
            call: self.tool_call.id,
            kind: AnswerKind::Executed(result),
        }
    }
}

/// A tool call whose arguments are not a JSON object. It has no parsed
/// arguments and is never executed; answer it with [`MalformedCall::answer`].
///
/// It cannot be answered as executed:
///
/// ```compile_fail,E0308
/// # use rig_agent::run::MalformedCall;
/// # use rig_core::tool::{ToolOutput, ToolResult};
/// fn answer(call: MalformedCall) {
///     let _ = call.answer(ToolResult::success(ToolOutput::text("wrote 5 bytes")));
/// }
/// ```
///
/// Nor read as a call to execute:
///
/// ```compile_fail,E0624
/// # use rig_agent::run::MalformedCall;
/// fn arguments(call: &MalformedCall) {
///     let _ = call.tool_call();
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MalformedCall {
    pub(super) turn: usize,
    pub(super) index: usize,
    pub(super) tool_call: ToolCall,
}

impl MalformedCall {
    /// The provider's call id.
    pub fn id(&self) -> &CallId {
        &self.tool_call.id
    }

    /// The tool the model called.
    pub fn name(&self) -> &ToolName {
        &self.tool_call.function.name
    }

    /// The arguments text exactly as the model sent it.
    pub fn raw_arguments(&self) -> &str {
        self.tool_call
            .function
            .invalid_arguments
            .as_deref()
            .unwrap_or_default()
    }

    pub(crate) fn tool_call(&self) -> &ToolCall {
        &self.tool_call
    }

    /// Answer this call with the invalid-call hook's action, or `None` when no
    /// hook decided. `None` and [`InvalidToolCallAction::Retry`] tell the model
    /// to call again, `Skip` answers with a skipped result, and `Stop`, `Fail`
    /// and `Repair` end the run when the answer is applied.
    pub fn answer(self, action: Option<InvalidToolCallAction>) -> ToolAnswer {
        ToolAnswer {
            turn: self.turn,
            index: self.index,
            call: self.tool_call.id,
            kind: AnswerKind::Malformed(action),
        }
    }
}

/// The answer to one pending call, built only by that call and applied with
/// [`AgentRun::answer`](super::AgentRun::answer).
///
/// A driver cannot build one itself:
///
/// ```compile_fail,E0451
/// # use rig_agent::run::ToolAnswer;
/// # use rig_core::message::CallId;
/// fn forge(call: CallId, other: ToolAnswer) -> ToolAnswer {
///     ToolAnswer { index: 0, call, ..other }
/// }
/// ```
///
/// Nor hand the run result content it built:
///
/// ```compile_fail,E0599
/// # use rig_agent::run::AgentRun;
/// # use rig_core::message::UserContent;
/// fn answer(run: &mut AgentRun, results: Vec<UserContent>) {
///     let _ = run.tool_results(results);
/// }
/// ```
#[derive(Debug)]
pub struct ToolAnswer {
    /// The model turn whose batch issued the call; ids alone may repeat
    /// across turns.
    pub(crate) turn: usize,
    pub(crate) index: usize,
    pub(crate) call: CallId,
    pub(crate) kind: AnswerKind,
}

impl ToolAnswer {
    /// Whether applying this answer ends the run.
    pub(crate) fn ends_run(&self) -> bool {
        matches!(
            self.kind,
            AnswerKind::Malformed(Some(
                InvalidToolCallAction::Stop { .. }
                    | InvalidToolCallAction::Fail
                    | InvalidToolCallAction::Repair { .. }
            ))
        )
    }
}

#[derive(Debug)]
pub(crate) enum AnswerKind {
    Executed(ToolResult),
    Malformed(Option<InvalidToolCallAction>),
}
