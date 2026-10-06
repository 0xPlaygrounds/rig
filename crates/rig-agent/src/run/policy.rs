//! Invalid tool-call diagnostics and recovery decisions exchanged with drivers.
//!
//! ```
//! use rig_agent::run::policy::InvalidToolCallAction;
//! let action = InvalidToolCallAction::retry("Use an advertised tool name.");
//! assert!(matches!(action, InvalidToolCallAction::Retry { .. }));
//! ```

use rig_core::message::{Message, ToolChoice};
use serde::{Deserialize, Serialize};

/// Why a model-emitted tool call cannot be dispatched as written.
///
/// Not every [`InvalidToolCallAction`] fits every reason: a name can be
/// repaired, argument text cannot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "reason", rename_all = "snake_case")]
#[non_exhaustive]
pub enum InvalidToolCallReason {
    /// The name is not an executable tool for this turn.
    UnknownTool,
    /// The tool exists, but the active tool choice does not allow it.
    DisallowedByToolChoice,
    /// The arguments are not a JSON object. The raw text is in
    /// [`InvalidToolCallContext::args`].
    MalformedArguments {
        /// What the JSON parser rejected.
        error: String,
    },
}

impl InvalidToolCallReason {
    /// The reason for arguments `raw` that are not a JSON object, with the
    /// parser's description of what it rejected.
    pub fn malformed_arguments(raw: &str) -> Self {
        Self::MalformedArguments {
            error: arguments_parse_error(raw),
        }
    }
}

/// What the JSON parser rejects in arguments `raw` that are not an object.
pub(crate) fn arguments_parse_error(raw: &str) -> String {
    match serde_json::from_str::<serde_json::Value>(raw) {
        Err(error) => error.to_string(),
        Ok(value) => format!("expected a JSON object, found {}", json_kind(&value)),
    }
}

fn json_kind(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "a boolean",
        serde_json::Value::Number(_) => "a number",
        serde_json::Value::String(_) => "a string",
        serde_json::Value::Array(_) => "an array",
        serde_json::Value::Object(_) => "an object",
    }
}

/// Diagnostics for an invalid model-emitted tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct InvalidToolCallContext {
    /// Name emitted by the model.
    pub tool_name: String,
    /// Durable tool-call id: the provider's when it issued one, else rig's
    /// minted handle. Absent only when no call object exists at all.
    pub tool_call_id: Option<rig_core::message::CallId>,
    /// Emitted JSON arguments, when present. For
    /// [`InvalidToolCallReason::MalformedArguments`] this is the raw text the
    /// model sent.
    pub args: Option<String>,
    /// Executable tools advertised for the turn.
    pub available_tools: Vec<String>,
    /// Tools permitted by the active tool choice.
    pub allowed_tools: Vec<String>,
    /// Active tool choice.
    pub tool_choice: Option<ToolChoice>,
    /// Diagnostic history including the rejected output.
    pub chat_history: Vec<Message>,
    /// Whether the call came from the streaming path.
    pub is_streaming: bool,
    /// Why the call cannot be dispatched as written.
    pub reason: InvalidToolCallReason,
}

/// How an accepted, tool-free model turn should be retried.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RetryRequest {
    /// Discard the rejected response and reuse the same prompt and preceding
    /// history with fresh request preparation.
    ///
    /// Completion-call hooks, retrieval, and dynamic tool resolution run again,
    /// so the resulting provider request may differ from the rejected attempt.
    Repeat,
    /// Preserve the rejected assistant response and append corrective feedback.
    Feedback(String),
}

/// Action for invalid-tool-call hooks and manual invalid-call resolution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum InvalidToolCallAction {
    /// Preserve fail-fast behavior.
    Fail,
    /// Retry the model with corrective feedback.
    Retry {
        /// Feedback appended for the retry.
        feedback: String,
    },
    /// Repair the emitted tool name.
    Repair {
        /// Replacement registered tool name.
        tool_name: String,
    },
    /// Treat the invalid call as skipped.
    Skip {
        /// Synthetic model feedback.
        reason: String,
    },
    /// Stop the run.
    Stop {
        /// Stop reason.
        reason: String,
    },
}

impl InvalidToolCallAction {
    /// Creates an action that preserves fail-fast invalid-call handling.
    pub fn fail() -> Self {
        Self::Fail
    }

    /// Creates an action that retries the model with corrective feedback.
    pub fn retry(feedback: impl Into<String>) -> Self {
        Self::Retry {
            feedback: feedback.into(),
        }
    }

    /// Creates an action that replaces the invalid tool name.
    pub fn repair(tool_name: impl Into<String>) -> Self {
        Self::Repair {
            tool_name: tool_name.into(),
        }
    }

    /// Creates an action that treats the invalid call as skipped.
    pub fn skip(reason: impl Into<String>) -> Self {
        Self::Skip {
            reason: reason.into(),
        }
    }

    /// Creates an action that stops the run with the supplied reason.
    pub fn stop(reason: impl Into<String>) -> Self {
        Self::Stop {
            reason: reason.into(),
        }
    }
}

#[cfg(test)]
mod tests;
