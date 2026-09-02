//! Decisions as data for invalid tool-call recovery: what the run tells a
//! driver about a call it could not accept, and what the driver answers.

use rig_core::message::{Message, ToolChoice};
use rig_core::streaming::BlockId;
use serde::{Deserialize, Serialize};

/// Diagnostics for an invalid model-emitted tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvalidToolCallContext {
    /// Name emitted by the model.
    pub tool_name: String,
    /// Durable tool-call id: the provider's when it issued one, else rig's
    /// minted handle. Absent only when no call object exists at all.
    pub tool_call_id: Option<String>,
    /// The stream block the call arrived under, when it streamed.
    pub block_id: Option<BlockId>,
    /// Emitted JSON arguments, when present.
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
