#![allow(dead_code)]
//! Scripted-model shims for this crate's tests: the familiar
//! `MockTurn`/`MockStreamEvent`/`MockCompletionModel` vocabulary rendered
//! onto [`crate::provider::MockScript`] behind
//! [`crate::provider::ProviderConfig::Mock`]. The shim is pure data — the
//! transport is the provider dispatcher, exactly as in production.

use rig_core::OneOrMany;
use rig_core::completion::{CompletionRequest, CompletionResponse, Usage};
use rig_core::message::{AssistantContent, Reasoning, Text, ToolCall, ToolFunction};
use rig_core::one_or_many::EmptyListError;
use rig_core::streaming::{StreamFinal, StreamedAssistantContent, ToolCallDeltaContent};

use crate::provider::{MockScript, ProviderConfig};

const MOCK_PROVIDER: &str = "mock";

/// A scripted non-streaming mock completion turn.
#[derive(Clone, Debug)]
pub struct MockTurn {
    choice: OneOrMany<AssistantContent>,
    usage: Usage,
    message_id: Option<String>,
}

impl MockTurn {
    /// Create a text response turn.
    pub fn text(text: impl Into<String>) -> Self {
        Self::from_content(AssistantContent::text(text.into()))
    }

    /// Create a tool-call response turn.
    pub fn tool_call(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        Self::from_content(AssistantContent::ToolCall(ToolCall::new(
            id.into(),
            ToolFunction::new(name.into(), arguments),
        )))
    }

    /// Create a response turn from one assistant content item.
    pub fn from_content(content: AssistantContent) -> Self {
        Self {
            choice: OneOrMany::one(content),
            usage: Usage::new(),
            message_id: None,
        }
    }

    /// Create a response turn from multiple assistant content items.
    pub fn from_contents(
        content: impl IntoIterator<Item = AssistantContent>,
    ) -> Result<Self, EmptyListError> {
        Ok(Self {
            choice: OneOrMany::many(content)?,
            usage: Usage::new(),
            message_id: None,
        })
    }

    /// Set the provider-assigned `call_id` on every tool call in this turn.
    pub fn with_call_id(mut self, call_id: impl Into<String>) -> Self {
        let call_id = call_id.into();
        let updated = self.choice.iter().cloned().map(|content| match content {
            AssistantContent::ToolCall(mut tool_call) => {
                tool_call.call_id = Some(call_id.clone());
                AssistantContent::ToolCall(tool_call)
            }
            other => other,
        });
        self.choice = OneOrMany::many(updated).expect("a turn always has content");
        self
    }

    /// Override usage for this turn.
    pub fn with_usage(mut self, usage: Usage) -> Self {
        self.usage = usage;
        self
    }

    /// Set a provider-assigned assistant message ID for this turn.
    pub fn with_message_id(mut self, message_id: impl Into<String>) -> Self {
        self.message_id = Some(message_id.into());
        self
    }

    fn into_completion_response(self) -> CompletionResponse {
        let mut completion = CompletionResponse::new(self.choice, self.usage, MOCK_PROVIDER);
        if let Some(message_id) = self.message_id {
            completion = completion.with_message_id(message_id);
        }
        completion
    }
}

/// A scripted streaming event, rendered into
/// [`StreamedAssistantContent`] items for [`MockScript::with_streams`].
///
/// `MessageId` has no streamed-content representation of its own, so it is
/// folded into the turn's terminal [`StreamFinal`] record.
#[derive(Clone, Debug)]
pub enum MockStreamEvent {
    /// Text chunk.
    Text(String),
    /// Complete tool call event.
    ToolCall {
        id: String,
        name: String,
        arguments: serde_json::Value,
    },
    /// Tool call delta event.
    ToolCallDelta {
        id: String,
        internal_call_id: String,
        content: ToolCallDeltaContent,
    },
    /// Complete reasoning event.
    Reasoning(Reasoning),
    /// Reasoning delta event.
    ReasoningDelta {
        id: Option<String>,
        reasoning: String,
    },
    /// Provider-assigned message ID (folded into the terminal record).
    MessageId(String),
    /// Provider-native output item that Rig does not model.
    Unknown(serde_json::Value),
    /// Final stream record carrying usage and metadata.
    FinalResponse(StreamFinal),
}

impl MockStreamEvent {
    /// Create a text chunk.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }

    /// Create a complete tool call event.
    pub fn tool_call(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        Self::ToolCall {
            id: id.into(),
            name: name.into(),
            arguments,
        }
    }

    /// Create a tool call name delta.
    pub fn tool_call_name_delta(
        id: impl Into<String>,
        internal_call_id: impl Into<String>,
        name: impl Into<String>,
    ) -> Self {
        Self::ToolCallDelta {
            id: id.into(),
            internal_call_id: internal_call_id.into(),
            content: ToolCallDeltaContent::Name(name.into()),
        }
    }

    /// Create a tool call arguments delta.
    pub fn tool_call_arguments_delta(
        id: impl Into<String>,
        internal_call_id: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self::ToolCallDelta {
            id: id.into(),
            internal_call_id: internal_call_id.into(),
            content: ToolCallDeltaContent::Delta(arguments.into()),
        }
    }

    /// Create a complete reasoning event.
    pub fn reasoning(reasoning: impl AsRef<str>) -> Self {
        Self::Reasoning(Reasoning::new(reasoning.as_ref()))
    }

    /// Create a reasoning delta event.
    pub fn reasoning_delta(id: Option<impl Into<String>>, reasoning: impl Into<String>) -> Self {
        Self::ReasoningDelta {
            id: id.map(Into::into),
            reasoning: reasoning.into(),
        }
    }

    /// Create a provider-assigned message ID event.
    pub fn message_id(id: impl Into<String>) -> Self {
        Self::MessageId(id.into())
    }

    /// Create an unmodeled provider output item.
    pub fn unknown(value: serde_json::Value) -> Self {
        Self::Unknown(value)
    }

    /// Create a final response event with usage.
    pub fn final_response(usage: Usage) -> Self {
        Self::FinalResponse(StreamFinal::new(MOCK_PROVIDER, usage))
    }

    /// Create a final response event with default zero usage.
    pub fn final_response_with_default_usage() -> Self {
        Self::final_response(Usage::new())
    }

    /// Create a final response event whose usage has only `total_tokens` set.
    pub fn final_response_with_total_tokens(total_tokens: u64) -> Self {
        let mut usage = Usage::new();
        usage.total_tokens = total_tokens;
        Self::final_response(usage)
    }

    /// Render one turn's events into streamed items, preserving order.
    ///
    /// A complete tool call reuses the internal call id of any delta that
    /// shares its provider id (so chunked emissions assemble into a single
    /// call), and otherwise mints a deterministic one. A `MessageId` event
    /// is folded into the turn's `StreamFinal` record.
    fn into_items(events: Vec<Self>) -> Vec<StreamedAssistantContent> {
        let message_id = events.iter().rev().find_map(|event| match event {
            Self::MessageId(id) => Some(id.clone()),
            _ => None,
        });
        let delta_internal_id = |call_id: &str| {
            events.iter().find_map(|event| match event {
                Self::ToolCallDelta {
                    id,
                    internal_call_id,
                    ..
                } if id == call_id => Some(internal_call_id.clone()),
                _ => None,
            })
        };
        events
            .iter()
            .filter_map(|event| match event.clone() {
                Self::Text(text) => Some(StreamedAssistantContent::Text(Text::new(text))),
                Self::ToolCall {
                    id,
                    name,
                    arguments,
                } => {
                    let internal_call_id =
                        delta_internal_id(&id).unwrap_or_else(|| format!("ic-{id}"));
                    Some(StreamedAssistantContent::ToolCall {
                        tool_call: ToolCall::new(id, ToolFunction::new(name, arguments)),
                        internal_call_id,
                    })
                }
                Self::ToolCallDelta {
                    id,
                    internal_call_id,
                    content,
                } => Some(StreamedAssistantContent::ToolCallDelta {
                    id,
                    internal_call_id,
                    content,
                }),
                Self::Reasoning(reasoning) => Some(StreamedAssistantContent::Reasoning(reasoning)),
                Self::ReasoningDelta { id, reasoning } => {
                    Some(StreamedAssistantContent::ReasoningDelta { id, reasoning })
                }
                Self::MessageId(_) => None,
                Self::Unknown(value) => Some(StreamedAssistantContent::Unknown(value)),
                Self::FinalResponse(mut final_record) => {
                    if final_record.message_id.is_none() {
                        final_record.message_id = message_id.clone();
                    }
                    Some(StreamedAssistantContent::Final(final_record))
                }
            })
            .collect()
    }
}

/// A cloneable scripted model handle: `clone` shares the underlying
/// [`MockScript`] (its cursor and request log), so a test can keep a probe
/// while the agent owns the provider configuration.
#[derive(Clone, Default)]
pub struct MockCompletionModel {
    script: MockScript,
}

impl MockCompletionModel {
    /// Create a mock model from scripted non-streaming turns.
    pub fn new(turns: impl IntoIterator<Item = MockTurn>) -> Self {
        Self::from_turns(turns)
    }

    /// Create a mock model that returns one text completion.
    pub fn text(text: impl Into<String>) -> Self {
        Self::from_turns([MockTurn::text(text)])
    }

    /// Create a mock model from scripted non-streaming turns.
    pub fn from_turns(turns: impl IntoIterator<Item = MockTurn>) -> Self {
        Self {
            script: MockScript::from_responses(
                turns
                    .into_iter()
                    .map(MockTurn::into_completion_response)
                    .collect(),
            ),
        }
    }

    /// Create a mock model from scripted streaming turns.
    pub fn from_stream_turns(
        stream_turns: impl IntoIterator<Item = impl IntoIterator<Item = MockStreamEvent>>,
    ) -> Self {
        Self {
            script: MockScript::from_responses(Vec::new()).with_streams(
                stream_turns
                    .into_iter()
                    .map(|turn| MockStreamEvent::into_items(turn.into_iter().collect()))
                    .collect(),
            ),
        }
    }

    /// The provider configuration an agent completes against.
    pub fn provider(&self) -> ProviderConfig {
        ProviderConfig::Mock(self.script.clone())
    }

    /// Return cloned requests received by this model's script.
    pub fn requests(&self) -> Vec<CompletionRequest> {
        self.script.requests()
    }

    /// Return the number of requests served by this model's script.
    pub fn request_count(&self) -> usize {
        self.script.calls()
    }
}
