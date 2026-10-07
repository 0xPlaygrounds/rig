//! Invalid tool-call diagnostics and recovery decisions exchanged with drivers.
//!
//! ```
//! use rig_agent::run::policy::InvalidToolCallAction;
//! let action = InvalidToolCallAction::retry("Use an advertised tool name.");
//! assert!(matches!(action, InvalidToolCallAction::Retry { .. }));
//! ```

use std::collections::BTreeSet;

use rig_core::message::{Message, ToolCall, ToolChoice, ToolName};
use serde::{Deserialize, Serialize};

use super::prepare::PrepareError;
use super::response::PromptError;

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

/// The tool policy of one model turn, built once by
/// [`prepare_request`](super::prepare::prepare_request) and read by every
/// invalid-call check of that turn. The allowed set is derived from the
/// executable tools, the choice and the output tool, and never persisted, so
/// it cannot disagree with the choice.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "TurnPolicyRepr")]
pub struct TurnPolicy {
    executable: BTreeSet<String>,
    tool_choice: Option<ToolChoice>,
    output_tool: Option<String>,
    #[serde(skip_serializing)]
    allowed: BTreeSet<String>,
}

#[derive(Deserialize)]
struct TurnPolicyRepr {
    executable: BTreeSet<String>,
    tool_choice: Option<ToolChoice>,
    output_tool: Option<String>,
}

impl TryFrom<TurnPolicyRepr> for TurnPolicy {
    type Error = PrepareError;

    fn try_from(repr: TurnPolicyRepr) -> Result<Self, Self::Error> {
        Self::new(repr.executable, repr.tool_choice, repr.output_tool)
    }
}

impl TurnPolicy {
    /// The policy for a turn advertising `executable` tools and, in Tool
    /// output mode, the synthetic `output_tool`, under `tool_choice`.
    ///
    /// # Errors
    /// [`PrepareError::Request`] when the choice cannot be honored: `Required`
    /// with no advertised tool, an empty `Specific`, or a `Specific` naming a
    /// tool that is not advertised.
    pub fn new(
        executable: BTreeSet<String>,
        tool_choice: Option<ToolChoice>,
        output_tool: Option<String>,
    ) -> Result<Self, PrepareError> {
        Self::resolve(executable, tool_choice, output_tool, None)
    }

    /// [`new`](Self::new) with the names advertised before a per-turn
    /// `active_tools` allow-list, which only sharpens the error message.
    pub(crate) fn resolve(
        executable: BTreeSet<String>,
        tool_choice: Option<ToolChoice>,
        output_tool: Option<String>,
        pre_filter: Option<&BTreeSet<String>>,
    ) -> Result<Self, PrepareError> {
        let output = output_tool.as_deref();
        let hint = |active_tools_caused: bool| {
            if active_tools_caused {
                " A per-turn `active_tools` allow-list narrowed the advertised tools this turn; \
                 set a compatible `tool_choice` in the same `RequestPatch`, or widen `active_tools`."
            } else {
                ""
            }
        };

        let mut allowed = match &tool_choice {
            Some(ToolChoice::Required) if executable.is_empty() && output.is_none() => {
                return Err(PrepareError::Request(format!(
                    "ToolChoice::Required forces the model to call a tool, but no tools are \
                     advertised this turn.{}",
                    hint(pre_filter.is_some_and(|pf| !pf.is_empty()))
                )));
            }
            None | Some(ToolChoice::Auto | ToolChoice::Required) => executable.clone(),
            Some(ToolChoice::None) => BTreeSet::new(),
            Some(ToolChoice::Specific { function_names }) => {
                if function_names.is_empty() {
                    return Err(PrepareError::Request(
                        "ToolChoice::Specific requires at least one function name".to_string(),
                    ));
                }
                let missing = function_names
                    .iter()
                    .map(ToolName::as_str)
                    .filter(|name| !executable.contains(*name) && Some(*name) != output)
                    .collect::<Vec<_>>();
                if !missing.is_empty() {
                    let advertised: Vec<_> = executable
                        .iter()
                        .map(String::as_str)
                        .chain(output)
                        .collect();
                    // Attribute missing names to filtering only if they existed before it.
                    return Err(PrepareError::Request(format!(
                        "ToolChoice::Specific requested tool names not advertised this turn: \
                         {missing:?}. Advertised: {advertised:?}.{}",
                        hint(pre_filter.is_some_and(|pf| missing.iter().any(|n| pf.contains(*n))))
                    )));
                }
                function_names.iter().map(ToString::to_string).collect()
            }
        };
        // The output tool is allowed, so its call is not invalid, though it
        // never executes.
        if let Some(name) = &output_tool {
            allowed.insert(name.clone());
        }

        Ok(Self {
            executable,
            tool_choice,
            output_tool,
            allowed,
        })
    }

    /// Names of the real, dispatchable tools advertised this turn.
    pub fn executable(&self) -> &BTreeSet<String> {
        &self.executable
    }

    /// Names the model may call without it being an invalid tool call: the
    /// executable tools narrowed by the tool choice, plus the output tool.
    pub fn allowed(&self) -> &BTreeSet<String> {
        &self.allowed
    }

    /// The turn's effective tool choice (patch over spec).
    pub fn tool_choice(&self) -> Option<&ToolChoice> {
        self.tool_choice.as_ref()
    }

    /// In Tool output mode, the synthetic output tool's name (allowed but
    /// never executable).
    pub fn output_tool(&self) -> Option<&str> {
        self.output_tool.as_deref()
    }

    /// Whether the turn's choice is [`ToolChoice::None`], under which no
    /// call may be skipped into the history.
    pub fn forbids_calls(&self) -> bool {
        matches!(self.tool_choice, Some(ToolChoice::None))
    }

    /// Whether a call to `name` is allowed this turn.
    pub fn allows(&self, name: &str) -> bool {
        self.allowed.contains(name)
    }

    /// The run's error for a call to `tool_name` this turn does not allow.
    pub(crate) fn unknown_call(
        &self,
        tool_name: String,
        chat_history: Vec<Message>,
    ) -> PromptError {
        PromptError::UnknownToolCall {
            tool_name,
            available_tools: self.executable.iter().cloned().collect(),
            allowed_tools: self.allowed.iter().cloned().collect(),
            chat_history,
        }
    }

    /// Why `call` is not allowed: its tool is not executable at all, or is
    /// executable but excluded by the tool choice.
    pub(crate) fn name_reason(&self, call: &ToolCall) -> InvalidToolCallReason {
        if self.executable.contains(call.function.name.as_str()) {
            InvalidToolCallReason::DisallowedByToolChoice
        } else {
            InvalidToolCallReason::UnknownTool
        }
    }

    /// The invalid-call context for `call` under this turn's policy, which
    /// supplies the advertised tools, the allowed tools and the tool choice.
    pub fn invalid_call_context(
        &self,
        call: &ToolCall,
        args: Option<String>,
        chat_history: Vec<Message>,
        is_streaming: bool,
        reason: InvalidToolCallReason,
    ) -> InvalidToolCallContext {
        InvalidToolCallContext {
            tool_name: call.function.name.to_string(),
            tool_call_id: Some(call.id.clone()),
            args,
            available_tools: self.executable.iter().cloned().collect(),
            allowed_tools: self.allowed.iter().cloned().collect(),
            tool_choice: self.tool_choice.clone(),
            chat_history,
            is_streaming,
            reason,
        }
    }
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
