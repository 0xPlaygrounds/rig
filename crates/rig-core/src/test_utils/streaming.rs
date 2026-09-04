//! Streaming helpers for [`MockCompletionModel`](super::MockCompletionModel).

use crate::{
    completion::{CompletionError, Usage},
    message::ReasoningContent,
    streaming::{StreamFinal, ToolCallEnd, UnparseableToolInput},
};

use crate::providers::internal::adapter::AdapterOutput;

/// Provider descriptor name reported by the test doubles.
pub const MOCK_PROVIDER: &str = "mock";

/// Build the terminal record the mock model yields, carrying `usage`.
pub fn mock_final(usage: Usage) -> StreamFinal {
    StreamFinal::new(MOCK_PROVIDER, usage)
}

/// Convert a fixture JSON value into canonical params: `null`/`{}` mean
/// "none", any other non-object is a scripting mistake surfaced as a stream
/// error.
fn fixture_additional_params(
    value: serde_json::Value,
) -> Result<Option<crate::message::AdditionalParams>, CompletionError> {
    crate::message::AdditionalParams::try_from_value(value).map_err(|other| {
        CompletionError::ProviderError(format!(
            "mock stream fixture `additional_params` must be a JSON object, got: {other}"
        ))
    })
}

/// Build a terminal record whose usage has only `total_tokens` set.
pub fn mock_final_with_total_tokens(total_tokens: u64) -> StreamFinal {
    let mut usage = Usage::new();
    usage.total_tokens = total_tokens;
    mock_final(usage)
}

/// Scripted streaming event yielded by [`MockCompletionModel`](super::MockCompletionModel).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
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
    /// Tool call name delta event.
    ToolCallNameDelta { id: String, name: String },
    /// Tool call arguments delta event.
    ToolCallArgumentsDelta { id: String, arguments: String },
    /// The end of a tool call streamed as deltas: the accumulator
    /// finalizes the fragments into the completed call, as a wire's
    /// step-stop does.
    ToolCallEnd { id: String },
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
/// minted renderings (`reasoning-0`, `block-3`, `output-1`, `tool-2`,
/// `text-0`) are the fixture syntax for a `BlockId::Minted` of that kind
/// and index, and anything else is a wire id. This is *fixture encoding*,
/// not provenance recovery: production code never parses an id string —
/// provenance travels in [`BlockId`](crate::streaming::BlockId) itself.
fn fixture_part_id(id: String) -> crate::streaming::BlockId {
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
    crate::streaming::BlockId::wire(id)
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
        Self::ToolCallNameDelta {
            id: id.into(),
            name: name.into(),
        }
    }

    /// Create a tool call arguments delta.
    pub fn tool_call_arguments_delta(id: impl Into<String>, arguments: impl Into<String>) -> Self {
        Self::ToolCallArgumentsDelta {
            id: id.into(),
            arguments: arguments.into(),
        }
    }

    /// Create the end of a tool call streamed as deltas.
    pub fn tool_call_end(id: impl Into<String>) -> Self {
        Self::ToolCallEnd { id: id.into() }
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

    /// Emit this scripted event as canonical stream events into `out` — the
    /// same helper adapters use, so a script speaks exactly the grammar a
    /// wire would.
    pub(crate) fn emit(self, out: &mut AdapterOutput) -> Result<(), CompletionError> {
        match self {
            Self::Text(text) => out.text(text),
            Self::TextStart {
                id,
                additional_params,
            } => out.text_start(
                fixture_part_id(id),
                additional_params
                    .map(fixture_additional_params)
                    .transpose()?
                    .flatten(),
            ),
            Self::TextAdditionalParams(additional_params) => {
                match fixture_additional_params(additional_params)? {
                    // The real variant is non-empty by construction; an empty
                    // fixture object is a scripting mistake, not a no-op.
                    None => {
                        return Err(CompletionError::ProviderError(
                            "mock stream fixture `TextAdditionalParams` carries no data — \
                             drop the event instead"
                                .to_string(),
                        ));
                    }
                    Some(params) => out.text_meta(params),
                }
            }
            Self::ToolCall {
                id,
                name,
                arguments,
                call_id,
            } => {
                let key = fixture_part_id(id.clone());
                // A wire-derived key doubles as the durable id (the common
                // case: providers key by the id the wire issued); minted
                // keys carry none.
                let mut end = ToolCallEnd::whole(name, arguments);
                if let Some(tool_id) = key.wire_str() {
                    end = end.with_tool_id(tool_id);
                }
                if let Some(call_id) = call_id {
                    end = end.with_call_id(call_id);
                }
                out.tool_call(key, end);
            }
            Self::ToolCallNameDelta { id, name } => out.tool_name(&fixture_part_id(id), name),
            Self::ToolCallArgumentsDelta { id, arguments } => {
                out.tool_arguments(&fixture_part_id(id), arguments)
            }
            Self::ToolCallEnd { id } => out.tool_end(
                fixture_part_id(id),
                ToolCallEnd::new(UnparseableToolInput::Error),
            ),
            Self::Reasoning { id, content } => {
                // Fixture syntax: a wire-shaped id is both the key and the
                // durable handle; a legacy minted rendering is a key only.
                let key = fixture_part_id(id.clone());
                let provider_id = key
                    .wire_str()
                    .and_then(|_| crate::streaming::non_empty_id(id));
                out.reasoning_block(key, provider_id, content);
            }
            Self::ReasoningDelta { id, reasoning } => {
                let key = fixture_part_id(id.clone());
                let provider_id = key
                    .wire_str()
                    .and_then(|_| crate::streaming::non_empty_id(id));
                out.reasoning_delta(&key, provider_id, reasoning);
            }
            Self::MessageId(id) => out.message_id(id),
            Self::Unknown(value) => out.unknown(value.into()),
            Self::FinalResponse(response) => {
                // The mock's terminal type is `StreamFinal` itself, so `raw`
                // is the scripted terminal serialized — the same capture
                // every real adapter performs.
                let raw = serde_json::to_value(&response)?;
                out.final_record(response.with_raw(raw));
            }
            Self::Error(error) => out.error(error.into_completion_error()),
        }
        Ok(())
    }
}
