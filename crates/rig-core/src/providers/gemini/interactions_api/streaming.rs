use serde::{Deserialize, Serialize};

use super::interactions_api_types::{
    Content, ContentDelta, FunctionCallContent, Interaction, InteractionSseEvent, InteractionUsage,
    Step, TextContent, TextDelta, ThoughtContent, ThoughtSignatureDelta, ThoughtSummaryContent,
    ThoughtSummaryDelta, map_interaction_status,
};
use crate::error::ProviderError;
use crate::operation::{CallFragment, Completion, Finish, IfMalformed, TextPart};
use crate::providers::gemini::streaming::shared_parts;
use crate::providers::internal::thoughts::Thoughts;
use crate::providers::internal::wire;
use crate::wire::{Decoder, Flow, Out, WireEvent, WireFrame};
use serde_json::{Map, Value};

/// Recognized Interactions SSE tags. Listed events must decode fully;
/// unlisted tags classify as unknown.
const KNOWN_EVENT_TYPES: &[&str] = &[
    "interaction.created",
    "interaction.completed",
    "interaction.status_update",
    "step.start",
    "step.delta",
    "step.stop",
    "error",
];

/// Classify an Interactions SSE frame by its `event_type` tag.
fn classify_interaction_frame(data: &str) -> WireEvent<InteractionSseEvent> {
    wire::classify_tagged_frame(data, "event_type", |event_type| {
        KNOWN_EVENT_TYPES.contains(&event_type)
    })
}

/// Whole-resource markers that prevent malformed SSE frames from decoding
/// as default interactions.
const INTERACTION_MARKER_KEYS: &[&str] = &["steps", "status", "usage", "object", "id"];

/// A decoded SSE event or whole unary interaction resource.
pub enum InteractionsEvent {
    /// One `event_type`-tagged streaming event.
    Sse(InteractionSseEvent),
    /// The whole interaction resource, as the unary reply delivers it.
    Whole(Interaction),
}

/// Classify a tagged SSE event, falling back to whole-resource classification
/// through [`wire::classify_or`].
fn classify_interactions_frame(data: &str) -> WireEvent<InteractionsEvent> {
    wire::classify_or(
        data,
        |data| classify_interaction_frame(data).map(InteractionsEvent::Sse),
        |data| {
            wire::classify_marker_keyed_frame::<Interaction>(data, INTERACTION_MARKER_KEYS)
                .map(InteractionsEvent::Whole)
        },
    )
}

/// Final metadata yielded by an Interactions streaming response.
#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct StreamingCompletionResponse {
    pub usage: Option<InteractionUsage>,
    pub interaction: Option<Interaction>,
    /// Resolved model identifier (e.g. `gemini-2.5-pro-preview-05-06`), extracted from
    /// `Interaction.model`. The Interactions API has no `FinishReason` field; use
    /// `interaction.status` for lifecycle state.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_version: Option<String>,
}

impl From<&StreamingCompletionResponse> for crate::completion::Usage {
    fn from(value: &StreamingCompletionResponse) -> crate::completion::Usage {
        value
            .usage
            .as_ref()
            .map(crate::completion::Usage::from)
            .unwrap_or_default()
    }
}

impl From<StreamingCompletionResponse> for crate::completion::Usage {
    fn from(value: StreamingCompletionResponse) -> crate::completion::Usage {
        (&value).into()
    }
}

/// The Gemini Interactions wire's decoder: one state machine for the whole
/// interaction and its stream of steps.
#[derive(Default)]
pub struct InteractionsDecoder<'id> {
    /// Thought boundaries inferred from content transitions and signatures.
    thoughts: Thoughts<'id>,
    /// The answer text part text extends.
    text: Option<TextPart<'id>>,
}

/// One content item as the decoder writes it.
enum Chunk {
    Thought {
        text: String,
        signature: Option<String>,
    },
    Text(String),
    Call {
        name: String,
        arguments: Option<Value>,
        id: Option<String>,
    },
    /// A content item the choice has no part for, kept verbatim on a text
    /// part's metadata.
    Raw(crate::message::AdditionalParams),
}

impl<'id> InteractionsDecoder<'id> {
    fn close_text(&mut self, out: &mut Out<'id, Completion>) {
        if let Some(part) = self.text.take() {
            out.close_text(part);
        }
    }

    /// Write one content item: thoughts, the boundary text or a call
    /// makes, text, then the call.
    fn write(&mut self, chunk: Chunk, out: &mut Out<'id, Completion>) -> Result<(), ProviderError> {
        match chunk {
            Chunk::Thought { text, signature } => {
                if !text.is_empty() {
                    self.close_text(out);
                }
                self.thoughts.fragment(out, &text);
                if let Some(signature) = signature {
                    self.thoughts.signature(out, signature);
                }
            }
            Chunk::Text(text) => {
                if text.is_empty() {
                    return Ok(());
                }
                self.thoughts.boundary();
                let part = self.text.get_or_insert_with(|| out.text());
                out.push_text(part, &text);
            }
            Chunk::Call {
                name,
                arguments,
                id,
            } => {
                self.thoughts.boundary();
                self.close_text(out);
                shared_parts::function_call(
                    out,
                    name,
                    arguments.unwrap_or(Value::Object(Map::new())),
                    id,
                    None,
                )?;
            }
            Chunk::Raw(params) => {
                self.thoughts.boundary();
                self.close_text(out);
                let part = out.text();
                out.text_params(&part, params);
                out.close_text(part);
            }
        }
        Ok(())
    }
}

/// EOF without `interaction.completed` is truncation, not successful
/// completion, so the decoder has nothing to add at the end of the reply.
impl<'id> Decoder<'id, Completion> for InteractionsDecoder<'id> {
    type Event = InteractionsEvent;

    fn classify(&self, frame: WireFrame) -> WireEvent<InteractionsEvent> {
        classify_interactions_frame(&frame.as_str())
    }

    fn decode(
        &mut self,
        event: InteractionsEvent,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        let event = match event {
            InteractionsEvent::Sse(event) => event,
            // The whole interaction states its content at once, in the
            // order a stream would write it.
            InteractionsEvent::Whole(interaction) => {
                for content in interaction.output_contents() {
                    if let Some(chunk) = content_chunk(content) {
                        self.write(chunk, &mut out)?;
                    }
                }
                InteractionSseEvent::InteractionCompleted {
                    interaction,
                    event_id: None,
                }
            }
        };

        match event {
            InteractionSseEvent::StepDelta { index, delta, .. } => match delta {
                ContentDelta::ArgumentsDelta(arguments_delta) => {
                    let index = index as usize;
                    if let Some(fragment) = arguments_delta.arguments
                        && !out.pending_name(index).is_empty()
                    {
                        out.call_fragment(
                            index,
                            CallFragment {
                                arguments: Some(fragment.as_str()),
                                ..CallFragment::default()
                            },
                        )?;
                    } else {
                        tracing::warn!(
                            step_index = index,
                            "arguments_delta with no open function-call step; dropping fragment"
                        );
                    }
                }
                ContentDelta::ThoughtSummary(ThoughtSummaryDelta { content }) => {
                    if let ThoughtSummaryContent::Text(text) = content {
                        self.write(
                            Chunk::Thought {
                                text: text.text,
                                signature: None,
                            },
                            &mut out,
                        )?;
                    }
                }
                ContentDelta::ThoughtSignature(ThoughtSignatureDelta { signature }) => {
                    // Signatures must survive even when no reasoning text streamed.
                    self.thoughts.signature(&mut out, signature);
                }
                delta => {
                    if let Some(chunk) = delta_content(delta).and_then(content_chunk) {
                        self.write(chunk, &mut out)?;
                    }
                }
            },
            InteractionSseEvent::StepStart { index, step, .. } => {
                if let Step::FunctionCall(FunctionCallContent {
                    name: Some(name),
                    arguments,
                    id,
                }) = step
                {
                    // The call stays open: its arguments may arrive in later
                    // deltas.
                    self.thoughts.boundary();
                    self.close_text(&mut out);
                    let index = index as usize;
                    out.call_fragment(
                        index,
                        CallFragment {
                            id: id.as_deref(),
                            name: Some(name.as_str()),
                            ..CallFragment::default()
                        },
                    )?;
                    // Announced arguments are a fallback, not an appendable
                    // fragment: combining them with later deltas could
                    // concatenate JSON objects.
                    if let Some(arguments) = arguments.filter(|arguments| {
                        arguments
                            .as_object()
                            .is_none_or(|object| !object.is_empty())
                    }) {
                        out.announce_pending(index, arguments);
                    }
                } else {
                    for chunk in step_start_chunks(step) {
                        self.write(chunk, &mut out)?;
                    }
                }
            }
            InteractionSseEvent::StepStop { index, .. } => {
                // A completed call with malformed arguments fails the reply.
                out.close_pending(index as usize, IfMalformed::Fail)?;
            }
            InteractionSseEvent::InteractionCompleted { interaction, .. } => {
                let span = tracing::Span::current();
                span.record("gen_ai.response.id", &interaction.id);
                if let Some(model) = interaction.model.clone() {
                    span.record("gen_ai.response.model", model);
                }
                // Provider completion finalizes calls even without step.stop.
                for index in out.pending_calls() {
                    tracing::debug!(
                        index,
                        "closing a function-call step left open at interaction.completed"
                    );
                    out.close_pending(index, IfMalformed::Fail)?;
                }
                self.close_text(&mut out);
                self.thoughts.close(&mut out, None);

                // Lifecycle status supplies the finish reason; absent status stays unknown.
                let model_version = interaction.model.clone();
                let native = StreamingCompletionResponse {
                    usage: interaction.usage,
                    interaction: Some(interaction),
                    model_version,
                };
                out.raw(serde_json::to_value(&native)?);
                let usage = (&native).into();
                let interaction = native.interaction.as_ref();
                let finish_reason = interaction
                    .and_then(|interaction| interaction.status.as_ref())
                    .map(map_interaction_status);
                let response_id = interaction
                    .map(|interaction| interaction.id.clone())
                    .filter(|id| !id.is_empty());
                return Ok(out.end(
                    Finish::new(usage)
                        .with_optional_reason(finish_reason)
                        .with_optional_response_id(response_id)
                        .with_optional_model(native.model_version),
                ));
            }
            event @ InteractionSseEvent::Error { .. } => {
                // Preserve modeled error fields without inventing an HTTP
                // status for an in-band failure.
                let body = serde_json::to_string(&event).unwrap_or_default();
                return Err(crate::error::ProviderError::from_provider_body(body));
            }
            InteractionSseEvent::InteractionCreated { .. }
            | InteractionSseEvent::InteractionStatusUpdate { .. } => {}
        }
        Ok(Flow::More)
    }
}

/// The content item a `step.delta` restates: a text or whole-call delta is
/// the item itself, so it takes the one content → block mapping. Every
/// other delta kind carries nothing the stream vocabulary models.
fn delta_content(delta: ContentDelta) -> Option<Content> {
    match delta {
        ContentDelta::Text(TextDelta { text, annotations }) => {
            text.map(|text| Content::Text(TextContent { text, annotations }))
        }
        ContentDelta::FunctionCall(call) => Some(Content::FunctionCall(call)),
        _ => None,
    }
}

fn step_start_chunks(step: Step) -> Vec<Chunk> {
    match step {
        // Model output can interleave multiple text and function-call items.
        Step::ModelOutput { content } => content.into_iter().filter_map(content_chunk).collect(),
        Step::FunctionCall(call) => content_chunk(Content::FunctionCall(call))
            .into_iter()
            .collect(),
        _ => Vec::new(),
    }
}

/// A supported output content item as the chunk it writes; other content is
/// skipped.
fn content_chunk(content: Content) -> Option<Chunk> {
    match content {
        Content::Text(text) if !text.text.is_empty() => Some(Chunk::Text(text.text)),
        Content::FunctionCall(FunctionCallContent {
            name,
            arguments,
            id,
        }) => Some(Chunk::Call {
            name: name?,
            arguments,
            id,
        }),
        // A thought the reply states whole: the summary's text is the
        // part's content and the signature closes it, the same pair the
        // streamed `thought_summary`/`thought_signature` deltas deliver
        // piecewise.
        Content::Thought(ThoughtContent {
            summary, signature, ..
        }) => {
            let text: String = summary
                .unwrap_or_default()
                .into_iter()
                .filter_map(|content| match content {
                    ThoughtSummaryContent::Text(text) => Some(text.text),
                    _ => None,
                })
                .collect();
            if text.is_empty() && signature.is_none() {
                return None;
            }
            Some(Chunk::Thought { text, signature })
        }
        // Images ride on a text part's metadata: the choice has no part for
        // them.
        image @ Content::Image(_) => crate::message::AdditionalParams::from_entries([(
            crate::providers::gemini::GEMINI_RAW_CONTENT_KEY,
            serde_json::json!(image),
        )])
        .map(Chunk::Raw),
        _ => None,
    }
}

#[cfg(test)]
mod tests;
