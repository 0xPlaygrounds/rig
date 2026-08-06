//! Streaming helpers for [`MockCompletionModel`](super::MockCompletionModel).

use crate::{
    completion::{CompletionError, Usage},
    message::ReasoningContent,
    streaming::{RawStreamingChoice, RawStreamingToolCall, StreamFinal, ToolCallDeltaContent},
};

/// Provider descriptor name reported by the test doubles.
pub const MOCK_PROVIDER: &str = "mock";

/// Build the terminal record the mock model yields, carrying `usage`.
pub fn mock_final(usage: Usage) -> StreamFinal {
    StreamFinal::new(MOCK_PROVIDER, usage)
}

/// Build a terminal record whose usage has only `total_tokens` set.
pub fn mock_final_with_total_tokens(total_tokens: u64) -> StreamFinal {
    let mut usage = Usage::new();
    usage.total_tokens = total_tokens;
    mock_final(usage)
}

/// Scripted streaming event yielded by [`MockCompletionModel`](super::MockCompletionModel).
#[derive(Clone, Debug)]
pub enum MockStreamEvent {
    /// Text chunk.
    Text(String),
    /// Start a new text content block with optional provider metadata.
    TextStart {
        id: String,
        additional_params: Option<serde_json::Value>,
    },
    /// Provider-specific metadata for the current text content block.
    TextAdditionalParams(serde_json::Value),
    /// Complete tool call event.
    ToolCall {
        id: String,
        name: String,
        arguments: serde_json::Value,
        call_id: Option<String>,
    },
    /// Tool call delta event.
    ToolCallDelta {
        id: String,
        content: ToolCallDeltaContent,
    },
    /// Complete reasoning event.
    Reasoning {
        id: String,
        content: ReasoningContent,
    },
    /// Reasoning delta event.
    ReasoningDelta { id: String, reasoning: String },
    /// Provider-assigned message ID.
    MessageId(String),
    /// Provider-native output item that Rig does not model.
    Unknown(serde_json::Value),
    /// Final raw response carrying optional usage.
    FinalResponse(StreamFinal),
    /// Stream error.
    Error(MockError),
}

use super::completion::MockError;

/// Fixture-syntax decoding of a part identity.
///
/// Corpus fixtures are plain data and spell identities as strings; the
/// legacy minted renderings (`reasoning-0`, `block-3`, `output-1`, `tool-2`,
/// `text-0`) are the fixture syntax for a [`PartId::Minted`] of that kind
/// and index, and anything else is a wire id. This is *fixture encoding*,
/// not provenance recovery: production code never parses an id string —
/// provenance travels in [`PartId`] itself.
fn fixture_part_id(id: String) -> crate::streaming::PartId {
    use crate::streaming::MintKind;
    for (namespace, kind) in [
        ("reasoning-", MintKind::Reasoning),
        ("block-", MintKind::Block),
        ("output-", MintKind::Output),
        ("tool-", MintKind::Tool),
        ("text-", MintKind::Text),
    ] {
        if let Some(rest) = id.strip_prefix(namespace)
            && let Ok(index) = rest.parse::<u64>()
        {
            return kind.for_wire_index(index);
        }
    }
    crate::streaming::PartId::wire(id)
}

impl MockStreamEvent {
    /// Create a text chunk.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }

    /// Start a new text content block identified by `id`.
    pub fn text_start(id: impl Into<String>, additional_params: Option<serde_json::Value>) -> Self {
        Self::TextStart {
            id: id.into(),
            additional_params,
        }
    }

    /// Add provider-specific metadata to the current text content block.
    pub fn text_additional_params(additional_params: serde_json::Value) -> Self {
        Self::TextAdditionalParams(additional_params)
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
            call_id: None,
        }
    }

    /// Attach a provider-specific call ID to a complete tool call event.
    pub fn with_call_id(mut self, call_id: impl Into<String>) -> Self {
        if let Self::ToolCall { call_id: id, .. } = &mut self {
            *id = Some(call_id.into());
        }
        self
    }

    /// Create a tool call name delta.
    pub fn tool_call_name_delta(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self::ToolCallDelta {
            id: id.into(),
            content: ToolCallDeltaContent::Name(name.into()),
        }
    }

    /// Create a tool call arguments delta.
    pub fn tool_call_arguments_delta(id: impl Into<String>, arguments: impl Into<String>) -> Self {
        Self::ToolCallDelta {
            id: id.into(),
            content: ToolCallDeltaContent::Delta(arguments.into()),
        }
    }

    /// Create a complete reasoning event with the default mock id
    /// (`"reasoning-0"`). Use [`Self::with_reasoning_id`] for tests that
    /// need distinct reasoning items.
    pub fn reasoning(reasoning: impl Into<String>) -> Self {
        Self::Reasoning {
            id: "reasoning-0".to_string(),
            content: ReasoningContent::Text {
                text: reasoning.into(),
                signature: None,
            },
        }
    }

    /// Attach a provider-specific reasoning ID to a complete reasoning event.
    pub fn with_reasoning_id(mut self, reasoning_id: impl Into<String>) -> Self {
        if let Self::Reasoning { id, .. } = &mut self {
            *id = reasoning_id.into();
        }
        self
    }

    /// Create a reasoning delta event with the default mock id
    /// (`"reasoning-0"`). Use [`Self::reasoning_delta_with_id`] for tests
    /// that need distinct reasoning items.
    pub fn reasoning_delta(reasoning: impl Into<String>) -> Self {
        Self::reasoning_delta_with_id("reasoning-0", reasoning)
    }

    /// Create a reasoning delta event with an explicit reasoning item id.
    pub fn reasoning_delta_with_id(id: impl Into<String>, reasoning: impl Into<String>) -> Self {
        Self::ReasoningDelta {
            id: id.into(),
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
        Self::FinalResponse(mock_final(usage))
    }

    /// Create a final response event with default zero usage.
    pub fn final_response_with_default_usage() -> Self {
        Self::FinalResponse(mock_final(Usage::new()))
    }

    /// Create a final response event whose usage has only `total_tokens` set.
    pub fn final_response_with_total_tokens(total_tokens: u64) -> Self {
        Self::FinalResponse(mock_final_with_total_tokens(total_tokens))
    }

    /// Create a stream error event.
    pub fn error(message: impl Into<String>) -> Self {
        Self::Error(MockError::provider(message))
    }

    pub(crate) fn into_raw_choice(self) -> Result<RawStreamingChoice, CompletionError> {
        match self {
            Self::Text(text) => Ok(RawStreamingChoice::Message(text)),
            Self::TextStart {
                id,
                additional_params,
            } => Ok(RawStreamingChoice::TextStart {
                id: fixture_part_id(id),
                additional_params,
            }),
            Self::TextAdditionalParams(additional_params) => {
                Ok(RawStreamingChoice::TextAdditionalParams(additional_params))
            }
            Self::ToolCall {
                id,
                name,
                arguments,
                call_id,
            } => {
                let mut tool_call = RawStreamingToolCall::new(fixture_part_id(id), name, arguments);
                if let Some(call_id) = call_id {
                    tool_call = tool_call.with_call_id(call_id);
                }
                Ok(RawStreamingChoice::ToolCall(tool_call))
            }
            Self::ToolCallDelta { id, content } => Ok(RawStreamingChoice::ToolCallDelta {
                id: fixture_part_id(id),
                content,
            }),
            Self::Reasoning { id, content } => Ok(RawStreamingChoice::Reasoning {
                id: fixture_part_id(id),
                content,
            }),
            Self::ReasoningDelta { id, reasoning } => Ok(RawStreamingChoice::ReasoningDelta {
                id: fixture_part_id(id),
                reasoning,
            }),
            Self::MessageId(id) => Ok(RawStreamingChoice::MessageId(id)),
            Self::Unknown(value) => Ok(RawStreamingChoice::Unknown(value)),
            Self::FinalResponse(response) => Ok(RawStreamingChoice::FinalResponse(response)),
            Self::Error(error) => Err(error.into_completion_error()),
        }
    }
}
