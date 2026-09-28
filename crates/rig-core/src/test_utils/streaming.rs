//! Streaming helpers for [`MockCompletionModel`](super::MockCompletionModel):
//! the script a mock reply is written from, and the decoder that writes it
//! through the completion writer's part handles, as a wire's decoder does.

use std::collections::HashMap;

use crate::completion::{CompletionResponse, Usage};
use crate::error::ProviderError;
use crate::message::ReasoningContent;
use crate::operation::{
    CallFragment, Completion, Finish, IfMalformed, ReasoningPart, Seal, TextPart,
};
use crate::wire::{Decoder, Flow, Out, WireEvent};

/// Provider descriptor name reported by the test doubles.
pub const MOCK_PROVIDER: &str = "mock";

/// The end the mock model's reply finishes with, carrying `usage`.
pub fn mock_final(usage: Usage) -> Finish {
    Finish::new(usage)
}

/// Convert a fixture JSON value into canonical params: `null`/`{}` mean
/// "none", any other non-object is a scripting mistake surfaced as a stream
/// error.
fn fixture_additional_params(
    value: serde_json::Value,
) -> Result<Option<crate::message::AdditionalParams>, ProviderError> {
    crate::message::AdditionalParams::try_from_value(value).map_err(|other| {
        ProviderError::Provider(format!(
            "mock stream fixture `additional_params` must be a JSON object, got: {other}"
        ))
    })
}

/// The end of a reply whose usage has only `total_tokens` set.
pub fn mock_final_with_total_tokens(total_tokens: u64) -> Finish {
    mock_final(Usage {
        total_tokens: Some(total_tokens),
        ..Default::default()
    })
}

/// Scripted streaming event yielded by [`MockCompletionModel`](super::MockCompletionModel).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum MockStreamEvent {
    /// Text chunk.
    Text(String),
    /// Start a new text part with optional provider metadata.
    TextStart {
        id: String,
        additional_params: Option<serde_json::Value>,
    },
    /// Provider-specific metadata for the current text part.
    TextAdditionalParams(serde_json::Value),
    /// Complete tool call event.
    ToolCall {
        id: String,
        name: String,
        arguments: serde_json::Value,
        call_id: Option<String>,
    },
    /// A tool call's name, as a wire streams it.
    ToolCallNameDelta { id: String, name: String },
    /// A fragment of a tool call's arguments.
    ToolCallArgumentsDelta { id: String, arguments: String },
    /// The end of a tool call streamed as fragments: the call closes with
    /// what they carried, as a wire's step-stop does.
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
    /// The provider's end of the reply.
    FinalResponse(Finish),
    /// A failure, which ends the reply.
    Error(MockError),
}

use super::completion::MockError;

/// The provider id a fixture spells, when it spells one. Corpus fixtures
/// are plain data: the renderings `reasoning-0`, `block-3`, `output-1`,
/// `tool-2` and `text-0` name a part the wire gave no id, and anything else
/// is the wire's own id.
fn fixture_provider_id(id: &str) -> Option<&str> {
    let unnamed = ["reasoning-", "block-", "output-", "tool-", "text-"]
        .iter()
        .any(|namespace| {
            id.strip_prefix(namespace)
                .is_some_and(|rest| rest.parse::<u64>().is_ok())
        });
    (!id.is_empty() && !unnamed).then_some(id)
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

    /// Create the provider's end of the reply, with usage.
    pub fn final_response(usage: Usage) -> Self {
        Self::FinalResponse(mock_final(usage))
    }

    /// Create a final response event whose usage reports no counter.
    pub fn final_response_with_default_usage() -> Self {
        Self::FinalResponse(mock_final(Usage::default()))
    }

    /// Create a final response event whose usage has only `total_tokens` set.
    pub fn final_response_with_total_tokens(total_tokens: u64) -> Self {
        Self::FinalResponse(mock_final_with_total_tokens(total_tokens))
    }

    /// Create a stream error event.
    pub fn error(message: impl Into<String>) -> Self {
        Self::Error(MockError::provider(message))
    }
}

/// One step of a mock reply: a scripted event, or a whole response.
#[derive(Clone, Debug)]
pub enum MockFrame {
    /// A scripted event.
    Event(MockStreamEvent),
    /// A whole response, as a unary turn answers.
    Response(Box<CompletionResponse>),
}

/// The decoder of [`MockScript`](super::MockScript): each scripted step is
/// written through the completion writer's part handles.
#[derive(Default)]
pub struct MockDecoder<'id> {
    /// The text part bare text extends.
    text: Option<TextPart<'id>>,
    /// The reasoning part each scripted id streams into, in start order.
    reasoning: Vec<(String, ReasoningPart<'id>)>,
    /// The buffer index each scripted call id streams under.
    calls: HashMap<String, usize>,
    next_call: usize,
}

impl<'id> MockDecoder<'id> {
    fn close_text(&mut self, out: &mut Out<'id, Completion>) {
        if let Some(part) = self.text.take() {
            out.close_text(part);
        }
    }

    fn call_index(&mut self, id: &str) -> usize {
        if let Some(index) = self.calls.get(id) {
            return *index;
        }
        let index = self.next_call;
        self.next_call += 1;
        if !id.is_empty() {
            self.calls.insert(id.to_owned(), index);
        }
        index
    }

    fn event(
        &mut self,
        event: MockStreamEvent,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        match event {
            MockStreamEvent::Text(text) => {
                let part = self.text.get_or_insert_with(|| out.text());
                out.push_text(part, &text);
            }
            MockStreamEvent::TextStart {
                id: _,
                additional_params,
            } => {
                self.close_text(&mut out);
                let part = out.text();
                if let Some(params) = additional_params
                    .map(fixture_additional_params)
                    .transpose()?
                    .flatten()
                {
                    out.text_params(&part, params);
                }
                self.text = Some(part);
            }
            MockStreamEvent::TextAdditionalParams(additional_params) => {
                // The real metadata is non-empty by construction; an empty
                // fixture object is a scripting mistake, not a no-op.
                let Some(params) = fixture_additional_params(additional_params)? else {
                    return Err(ProviderError::Provider(
                        "mock stream fixture `TextAdditionalParams` carries no data — \
                         drop the event instead"
                            .to_string(),
                    ));
                };
                let part = self.text.get_or_insert_with(|| out.text());
                out.text_params(part, params);
            }
            MockStreamEvent::ToolCall {
                id,
                name,
                arguments,
                call_id,
            } => {
                self.close_text(&mut out);
                // An id-less call is a wire that sends none: rig issues the
                // id. A call id beside the wire id makes the pair a
                // dual-identifier call. A call scripted as fragments under
                // this id is the one this restatement closes.
                let index = match self.calls.remove(&id) {
                    Some(index) => index,
                    None => self.call_index(""),
                };
                let wire_id = fixture_provider_id(&id);
                out.call_fragment(
                    index,
                    CallFragment {
                        id: call_id.as_deref().or(wire_id),
                        item_id: call_id.as_ref().and(wire_id),
                        name: Some(name.as_str()),
                        ..CallFragment::default()
                    },
                )?;
                out.announce_pending(index, arguments);
                out.close_pending(index, IfMalformed::Fail)?;
            }
            MockStreamEvent::ToolCallNameDelta { id, name } => {
                self.close_text(&mut out);
                let index = self.call_index(&id);
                out.call_fragment(
                    index,
                    CallFragment {
                        id: fixture_provider_id(&id),
                        name: Some(name.as_str()),
                        ..CallFragment::default()
                    },
                )?;
            }
            MockStreamEvent::ToolCallArgumentsDelta { id, arguments } => {
                self.close_text(&mut out);
                let index = self.call_index(&id);
                out.call_fragment(
                    index,
                    CallFragment {
                        id: fixture_provider_id(&id),
                        arguments: Some(arguments.as_str()),
                        ..CallFragment::default()
                    },
                )?;
            }
            MockStreamEvent::ToolCallEnd { id } => {
                let index = self.call_index(&id);
                self.calls.remove(&id);
                out.close_pending(index, IfMalformed::Fail)?;
            }
            MockStreamEvent::Reasoning { id, content } => {
                self.close_text(&mut out);
                let reasoning = crate::message::Reasoning {
                    id: fixture_provider_id(&id).map(str::to_owned),
                    content: vec![content],
                };
                // A whole reasoning restates the part streamed under its id,
                // and closes it.
                match self.reasoning.iter().position(|(open, _)| *open == id) {
                    Some(at) => {
                        let (_, part) = self.reasoning.remove(at);
                        out.close_reasoning(
                            part,
                            Seal {
                                id: reasoning.id.clone(),
                                restated: Some(reasoning),
                                ..Seal::default()
                            },
                        );
                    }
                    None => out.reasoning_block(reasoning),
                }
            }
            MockStreamEvent::ReasoningDelta { id, reasoning } => {
                self.close_text(&mut out);
                let at = match self.reasoning.iter().position(|(open, _)| *open == id) {
                    Some(at) => at,
                    None => {
                        self.reasoning.push((id, out.reasoning()));
                        self.reasoning.len() - 1
                    }
                };
                if let Some((_, part)) = self.reasoning.get(at) {
                    out.push_reasoning(part, &reasoning);
                }
            }
            MockStreamEvent::MessageId(id) => out.message_id(id),
            MockStreamEvent::Unknown(value) => out.unknown(value.into()),
            MockStreamEvent::FinalResponse(finish) => {
                self.close_text(&mut out);
                for (id, part) in std::mem::take(&mut self.reasoning) {
                    out.close_reasoning(
                        part,
                        Seal {
                            id: fixture_provider_id(&id).map(str::to_owned),
                            ..Seal::default()
                        },
                    );
                }
                // The mock's document is its scripted end, serialized.
                out.raw(serde_json::to_value(&finish)?);
                return Ok(out.end(finish));
            }
            MockStreamEvent::Error(error) => return Err(error.into_completion_error()),
        }
        Ok(Flow::More)
    }

    fn response(
        &mut self,
        response: CompletionResponse,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        for content in response.choice.iter().cloned() {
            out.content(content)?;
        }
        if let Some(message_id) = response.message_id.clone() {
            out.message_id(message_id);
        }
        out.raw(response.raw.clone());
        Ok(out.end(
            Finish::new(response.usage)
                .with_optional_reason(response.finish_reason())
                .with_optional_response_id(response.response_id.clone())
                .with_optional_model(response.model.clone())
                .with_optional_provider_request_id(response.provider_request_id.clone()),
        ))
    }
}

impl<'id> Decoder<'id, Completion, MockFrame> for MockDecoder<'id> {
    type Event = MockFrame;

    fn classify(&self, frame: MockFrame) -> WireEvent<MockFrame> {
        WireEvent::Known(frame)
    }

    fn decode(
        &mut self,
        frame: MockFrame,
        out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        match frame {
            MockFrame::Event(event) => self.event(event, out),
            MockFrame::Response(response) => self.response(*response, out),
        }
    }
}
