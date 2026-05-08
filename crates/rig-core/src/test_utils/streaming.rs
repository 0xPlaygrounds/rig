//! Streaming helpers for [`MockCompletionModel`](super::MockCompletionModel).

use crate::{
    completion::{CompletionError, GetTokenUsage, Usage},
    streaming::{RawStreamingChoice, RawStreamingToolCall, ToolCallDeltaContent},
};
use serde::{Deserialize, Serialize};

/// Raw mock response used by completion and streaming test utilities.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct MockResponse {
    usage: Option<Usage>,
}

impl MockResponse {
    /// Create a mock raw response without token usage.
    pub fn new() -> Self {
        Self { usage: None }
    }

    /// Create a mock raw response carrying token usage.
    pub fn with_usage(usage: Usage) -> Self {
        Self { usage: Some(usage) }
    }

    /// Create a mock raw response whose usage has only `total_tokens` set.
    pub fn with_total_tokens(total_tokens: u64) -> Self {
        let mut usage = Usage::new();
        usage.total_tokens = total_tokens;
        Self::with_usage(usage)
    }
}

impl GetTokenUsage for MockResponse {
    fn token_usage(&self) -> Option<Usage> {
        self.usage
    }
}

/// Scripted streaming event yielded by [`MockCompletionModel`](super::MockCompletionModel).
#[derive(Clone, Debug)]
pub enum MockStreamEvent {
    /// Text chunk.
    Text(String),
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
        internal_call_id: String,
        content: ToolCallDeltaContent,
    },
    /// Provider-assigned message ID.
    MessageId(String),
    /// Final raw response carrying optional usage.
    FinalResponse(MockResponse),
    /// Stream error.
    Error(MockError),
}

use super::completion::MockError;

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

    /// Create a provider-assigned message ID event.
    pub fn message_id(id: impl Into<String>) -> Self {
        Self::MessageId(id.into())
    }

    /// Create a final response event with usage.
    pub fn final_response(usage: Usage) -> Self {
        Self::FinalResponse(MockResponse::with_usage(usage))
    }

    /// Create a final response event with default zero usage.
    pub fn final_response_with_default_usage() -> Self {
        Self::FinalResponse(MockResponse::with_usage(Usage::new()))
    }

    /// Create a final response event whose usage has only `total_tokens` set.
    pub fn final_response_with_total_tokens(total_tokens: u64) -> Self {
        Self::FinalResponse(MockResponse::with_total_tokens(total_tokens))
    }

    /// Create a stream error event.
    pub fn error(message: impl Into<String>) -> Self {
        Self::Error(MockError::provider(message))
    }

    pub(crate) fn into_raw_choice(
        self,
    ) -> Result<RawStreamingChoice<MockResponse>, CompletionError> {
        match self {
            Self::Text(text) => Ok(RawStreamingChoice::Message(text)),
            Self::ToolCall {
                id,
                name,
                arguments,
                call_id,
            } => {
                let mut tool_call = RawStreamingToolCall::new(id, name, arguments);
                if let Some(call_id) = call_id {
                    tool_call = tool_call.with_call_id(call_id);
                }
                Ok(RawStreamingChoice::ToolCall(tool_call))
            }
            Self::ToolCallDelta {
                id,
                internal_call_id,
                content,
            } => Ok(RawStreamingChoice::ToolCallDelta {
                id,
                internal_call_id,
                content,
            }),
            Self::MessageId(id) => Ok(RawStreamingChoice::MessageId(id)),
            Self::FinalResponse(response) => Ok(RawStreamingChoice::FinalResponse(response)),
            Self::Error(error) => Err(error.into_completion_error()),
        }
    }
}
