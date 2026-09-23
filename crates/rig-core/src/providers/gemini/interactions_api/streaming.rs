use serde::{Deserialize, Serialize};

use super::PROVIDER_NAME;
use super::interactions_api_types::{
    Content, ContentDelta, FunctionCallContent, Interaction, InteractionSseEvent, InteractionUsage,
    Step, TextContent, TextDelta, ThoughtContent, ThoughtSignatureDelta, ThoughtSummaryContent,
    ThoughtSummaryDelta, map_interaction_status,
};
use crate::providers::gemini::streaming::shared_parts;
use crate::providers::internal::chunk_lifecycle::ChunkParts;
use crate::providers::internal::tool_call_bridge::ToolCallBridge;

use crate::operation::Completion;
use crate::providers::internal::wire::{self, WireEvent};
use crate::streaming;
use crate::wire::WireFrame;
use crate::wire::{Decoder, Output};
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

/// The Gemini Interactions wire's decoder, for both of its modes.
///
/// Holds the per-reply state (thought lifecycle, open function-call step
/// assemblies); frame-triage policy is the driver's, not this decoder's.
pub struct InteractionsDecoder {
    /// Thought boundaries inferred from content transitions and signatures.
    reasoning: crate::providers::internal::chunk_lifecycle::MintedReasoningLifecycle,
    /// A provider error ended the turn; later frames must not produce output.
    failed: bool,
    /// Open function calls keyed by step index, retaining argument deltas.
    /// Its id minter is shared with whole calls to prevent local identity collisions.
    open_function_steps: ToolCallBridge<u32>,
}

impl Default for InteractionsDecoder {
    fn default() -> Self {
        Self {
            reasoning: crate::providers::internal::chunk_lifecycle::MintedReasoningLifecycle::new(
                crate::streaming::MintKind::Reasoning,
            ),
            failed: false,
            open_function_steps: ToolCallBridge::new(),
        }
    }
}

impl Decoder<Completion> for InteractionsDecoder {
    type Event = InteractionsEvent;

    fn classify(&self, frame: WireFrame) -> WireEvent<InteractionsEvent> {
        classify_interactions_frame(&frame.as_str())
    }

    fn interpret(&mut self, event: InteractionsEvent, out: &mut Output<Completion>) {
        if self.failed {
            return;
        }

        let event = match event {
            InteractionsEvent::Sse(event) => event,
            // Unary content uses the same lifecycle so block ordering matches streaming.
            InteractionsEvent::Whole(interaction) => {
                for content in interaction.output_contents() {
                    if let Some(parts) =
                        content_to_parts(content, self.open_function_steps.minted_ids())
                    {
                        self.reasoning.emit_chunk(parts, out);
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
                    if let (Some(slot), Some(fragment)) = (
                        self.open_function_steps.get_mut(index),
                        arguments_delta.arguments,
                    ) {
                        slot.saw_arguments_delta = true;
                        let key = slot.key().clone();
                        out.tool_arguments(&key, fragment);
                    } else {
                        tracing::warn!(
                            step_index = index,
                            "arguments_delta with no open function-call step; dropping fragment"
                        );
                    }
                }
                ContentDelta::ThoughtSummary(ThoughtSummaryDelta { content }) => {
                    if let ThoughtSummaryContent::Text(text) = content {
                        self.reasoning.emit_chunk(
                            ChunkParts {
                                reasoning: Some(text.text),
                                reasoning_signature: None,
                                text: None,
                                text_meta: None,
                                tool_events: Vec::new(),
                            },
                            out,
                        );
                    }
                }
                ContentDelta::ThoughtSignature(ThoughtSignatureDelta { signature }) => {
                    // Signatures must survive even when no reasoning text streamed.
                    self.reasoning.emit_chunk(
                        ChunkParts {
                            reasoning: None,
                            reasoning_signature: Some(signature),
                            text: None,
                            text_meta: None,
                            tool_events: Vec::new(),
                        },
                        out,
                    );
                }
                delta => {
                    if let Some(parts) = delta_content(delta).and_then(|content| {
                        content_to_parts(content, self.open_function_steps.minted_ids())
                    }) {
                        // Interleaving content must close any open thought block.
                        self.reasoning.emit_chunk(parts, out);
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
                    // Keep the call open because its arguments may arrive in later deltas.
                    let slot = self
                        .open_function_steps
                        .open(index, id.as_deref(), Some(&name));
                    // Announcement arguments are a fallback, not an appendable fragment;
                    // combining them with later deltas could concatenate JSON objects.
                    slot.announce_arguments = arguments.filter(|arguments| {
                        arguments
                            .as_object()
                            .is_none_or(|object| !object.is_empty())
                    });
                    let key = slot.key().clone();
                    let tool_events = vec![
                        streaming::StreamEvent::BlockStart {
                            id: key.clone(),
                            kind: streaming::BlockKind::ToolCall,
                        },
                        streaming::StreamEvent::BlockDelta {
                            id: key,
                            delta: streaming::Delta::ToolName { name },
                        },
                    ];
                    // Tool content interleaving an open thought block: the
                    // shared lifecycle synthesizes the boundary end.
                    self.reasoning.emit_chunk(
                        ChunkParts {
                            reasoning: None,
                            reasoning_signature: None,
                            text: None,
                            text_meta: None,
                            tool_events,
                        },
                        out,
                    );
                } else {
                    // Separate chunks preserve text/tool ordering and thought boundaries.
                    for parts in step_start_to_parts(step, self.open_function_steps.minted_ids()) {
                        self.reasoning.emit_chunk(parts, out);
                    }
                }
            }
            InteractionSseEvent::StepStop { index, .. } => {
                // A completed call with malformed arguments must fail in-band.
                if let Some(slot) = self.open_function_steps.remove(index) {
                    out.push(Ok(function_step_end(&slot)));
                }
            }
            InteractionSseEvent::InteractionCompleted { interaction, .. } => {
                let span = tracing::Span::current();
                span.record("gen_ai.response.id", &interaction.id);
                if let Some(model) = interaction.model.clone() {
                    span.record("gen_ai.response.model", model);
                }
                // Provider completion finalizes calls even without step.stop.
                // Announcement order keeps parallel call output deterministic.
                for (index, slot) in self.open_function_steps.drain_ordered_indexed() {
                    tracing::debug!(
                        index,
                        "closing a function-call step left open at interaction.completed"
                    );
                    out.push(Ok(function_step_end(&slot)));
                }

                // Lifecycle status supplies the finish reason; absent status stays unknown.
                let model_version = interaction.model.clone();
                let native = StreamingCompletionResponse {
                    usage: interaction.usage,
                    interaction: Some(interaction),
                    model_version,
                };
                let raw = match serde_json::to_value(&native) {
                    Ok(raw) => raw,
                    Err(err) => {
                        out.error(err.into());
                        return;
                    }
                };
                let usage = (&native).into();
                let interaction = native.interaction.as_ref();
                let finish_reason = interaction
                    .and_then(|interaction| interaction.status.as_ref())
                    .map(map_interaction_status);
                let message_id = interaction
                    .map(|interaction| interaction.id.as_str())
                    .filter(|id| !id.is_empty());
                out.final_record(
                    streaming::StreamFinal::new(PROVIDER_NAME, usage, raw)
                        .with_optional_finish_reason(finish_reason)
                        .with_optional_response_id(message_id)
                        .with_optional_model(native.model_version.as_deref()),
                );
            }
            event @ InteractionSseEvent::Error { .. } => {
                // Preserve modeled error fields without inventing an HTTP status
                // for an in-band failure.
                self.failed = true;
                let body = serde_json::to_string(&event).unwrap_or_default();
                out.push(Err(crate::error::ProviderError::from_provider_body(body)));
            }
            InteractionSseEvent::InteractionCreated { .. }
            | InteractionSseEvent::InteractionStatusUpdate { .. } => {}
        }
    }

    fn finish(&mut self, _out: &mut Output<Completion>) {
        // EOF without interaction.completed is truncation, not successful completion.
    }

    fn is_finished(&self) -> bool {
        // Stop after terminal errors so later unknown frames cannot escape the failure gate.
        self.failed
    }
}

/// Close a function call using accumulated arguments, announcement fallback,
/// or `{}` if neither exists. Malformed arguments fail in-band.
/// Preserve the provider id as `tool_id` only.
fn function_step_end(
    slot: &crate::providers::internal::tool_call_bridge::ToolCallSlot,
) -> streaming::StreamEvent {
    slot.end_event(streaming::UnparseableToolInput::Error)
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

/// A whole function call as one declared chunk (its start and end in the
/// tool-event slot).
fn function_call_parts(
    name: String,
    arguments: Option<Value>,
    id: Option<String>,
    tool_ids: &mut streaming::SyntheticIds,
) -> ChunkParts {
    // Tool names cannot identify calls because a turn may call the same tool twice.
    ChunkParts {
        reasoning: None,
        reasoning_signature: None,
        text: None,
        text_meta: None,
        tool_events: shared_parts::function_call(
            name,
            arguments.unwrap_or(Value::Object(Map::new())),
            id,
            None,
            tool_ids,
        ),
    }
}

/// Visible text as one declared chunk.
fn text_parts(text: String) -> ChunkParts {
    ChunkParts {
        reasoning: None,
        reasoning_signature: None,
        text: Some(text),
        text_meta: None,
        tool_events: Vec::new(),
    }
}

fn step_start_to_parts(step: Step, tool_ids: &mut streaming::SyntheticIds) -> Vec<ChunkParts> {
    match step {
        // Model output can interleave multiple text and function-call items.
        Step::ModelOutput { content } => content
            .into_iter()
            .filter_map(|content| content_to_parts(content, tool_ids))
            .collect(),
        Step::FunctionCall(call) => content_to_parts(Content::FunctionCall(call), tool_ids)
            .into_iter()
            .collect(),
        _ => Vec::new(),
    }
}

/// Convert supported output content into a canonical chunk; skip other content.
fn content_to_parts(
    content: Content,
    tool_ids: &mut streaming::SyntheticIds,
) -> Option<ChunkParts> {
    match content {
        Content::Text(text) if !text.text.is_empty() => Some(text_parts(text.text)),
        Content::FunctionCall(FunctionCallContent {
            name,
            arguments,
            id,
        }) => Some(function_call_parts(name?, arguments, id, tool_ids)),
        // A thought the reply states whole: the summary's text is the
        // block's content and the signature closes it, which is the same
        // pair the streamed `thought_summary`/`thought_signature` deltas
        // deliver piecewise.
        Content::Thought(ThoughtContent {
            summary, signature, ..
        }) => {
            let reasoning: String = summary
                .unwrap_or_default()
                .into_iter()
                .filter_map(|content| match content {
                    ThoughtSummaryContent::Text(text) => Some(text.text),
                    _ => None,
                })
                .collect();
            if reasoning.is_empty() && signature.is_none() {
                return None;
            }
            Some(ChunkParts {
                reasoning: (!reasoning.is_empty()).then_some(reasoning),
                reasoning_signature: signature,
                text: None,
                text_meta: None,
                tool_events: Vec::new(),
            })
        }
        // Preserve images in metadata because canonical stream blocks cannot represent them.
        image @ Content::Image(_) => raw_content_parts(image, tool_ids),
        _ => None,
    }
}

/// A content item the stream vocabulary has no block kind for, preserved as
/// a text block carrying the part verbatim under
/// [`GEMINI_RAW_CONTENT_KEY`](crate::providers::gemini::GEMINI_RAW_CONTENT_KEY).
fn raw_content_parts(
    content: Content,
    tool_ids: &mut streaming::SyntheticIds,
) -> Option<ChunkParts> {
    let params = crate::message::AdditionalParams::from_entries([(
        crate::providers::gemini::GEMINI_RAW_CONTENT_KEY,
        serde_json::json!(content),
    )])?;
    // Keyed from the same counter every id-less block on this wire draws
    // from, so a raw block can never collide with a minted tool-call key.
    let id = tool_ids.mint();
    Some(ChunkParts {
        reasoning: None,
        reasoning_signature: None,
        text: None,
        text_meta: None,
        tool_events: vec![
            streaming::StreamEvent::BlockStart {
                id: id.clone(),
                kind: streaming::BlockKind::Text {
                    additional_params: Some(params),
                },
            },
            streaming::StreamEvent::BlockEnd {
                id,
                end: streaming::BlockClose::Text,
                block: None,
            },
        ],
    })
}

#[cfg(test)]
mod tests;
