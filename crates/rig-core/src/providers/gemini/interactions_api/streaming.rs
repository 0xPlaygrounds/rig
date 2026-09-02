use async_stream::stream;
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::pin::Pin;

use super::interactions_api_types::{
    Content, ContentDelta, FunctionCallContent, Interaction, InteractionSseEvent, InteractionUsage,
    Step, TextDelta, ThoughtSignatureDelta, ThoughtSummaryContent, ThoughtSummaryDelta,
    map_interaction_status,
};
use super::{InteractionsCompletionModel, PROVIDER_NAME, create_request_body};
use crate::completion::{CompletionError, CompletionRequest};
use crate::http_client::HttpClientExt;
use crate::http_client::Request;
use crate::http_client::sse::{Event, GenericEventSource};
use crate::providers::gemini::streaming::shared_parts;
use crate::providers::internal::chunk_lifecycle::ChunkParts;
use crate::providers::internal::sse_transport::{
    OpenLog, SseTransportOptions, open_wire_stream, skip_blank_frames,
};
use crate::providers::internal::tool_call_bridge::ToolCallBridge;

use crate::providers::internal::adapter::{
    AdapterOutput, TriagedFrame, WireAdapter, WireFrame, triage_frame,
};
use crate::providers::internal::wire::{self, WireEvent};
use crate::streaming;
use crate::telemetry::{CompletionOperation, CompletionSpanBuilder, SpanCombinator};
use serde_json::{Map, Value};

/// The `event_type` values this client models on the Interactions SSE wire.
///
/// [`wire::classify_tagged_frame`] dispatches on this list: a frame whose
/// `event_type` is outside it classifies `Unknown` (driver policy: warn +
/// skip), while a listed value must pass the full [`InteractionSseEvent`]
/// decode or classify `Corrupt`. There is no untagged serde fallback — policy
/// lives in the classify layer, never in serde.
const KNOWN_EVENT_TYPES: &[&str] = &[
    "interaction.created",
    "interaction.completed",
    "interaction.status_update",
    "step.start",
    "step.delta",
    "step.stop",
    "error",
];

/// Classify one Interactions SSE frame. The single classify site for both
/// consumers of this wire: the completion adapter below and the raw
/// [`stream_interaction_events`] surface.
fn classify_interaction_frame(data: &str) -> WireEvent<InteractionSseEvent> {
    wire::classify_tagged_frame(data, "event_type", |event_type| {
        KNOWN_EVENT_TYPES.contains(&event_type)
    })
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

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub type InteractionEventStream =
    Pin<Box<dyn Stream<Item = Result<InteractionSseEvent, CompletionError>> + Send>>;

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub type InteractionEventStream =
    Pin<Box<dyn Stream<Item = Result<InteractionSseEvent, CompletionError>>>>;

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

impl<T> InteractionsCompletionModel<T>
where
    T: HttpClientExt + Clone + 'static,
{
    pub(crate) async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<streaming::StreamingCompletionResponse, CompletionError> {
        let span = CompletionSpanBuilder::new(
            PROVIDER_NAME,
            &self.model,
            CompletionOperation::InteractionsStreaming,
        )
        .system_instructions(
            completion_request.system_instructions(),
            completion_request.record_telemetry_content,
        )
        .build();

        let request = create_request_body(self.model.clone(), completion_request, Some(true))?;

        crate::providers::internal::trace_json(
            crate::providers::internal::LogTarget::Streaming,
            "Gemini interactions streaming request",
            &request,
        );

        let body = serde_json::to_vec(&request)?;
        let req = self
            .client
            .post("/v1beta/interactions?alt=sse")?
            .header("Content-Type", "application/json")
            .body(body)
            .map_err(|e| CompletionError::HttpError(e.into()))?;

        Ok(streaming::StreamingCompletionResponse::stream(
            PROVIDER_NAME,
            open_wire_stream(
                GenericEventSource::new(self.client.clone(), req),
                SseTransportOptions {
                    open_log: OpenLog::Debug,
                    stream_ended_is_error: false,
                    log_transport_errors: true,
                },
                skip_blank_frames,
                InteractionsAdapter::default(),
                span,
            ),
        ))
    }
}

/// The Gemini Interactions SSE wire as a [`WireAdapter`].
///
/// Frame-triage policy (warn on `Unknown`, in-band `Err` on `Corrupt`) lives
/// in [`run_wire_stream`], not here — this ends the wire's former
/// debug-log-and-skip handling of every decode failure.
struct InteractionsAdapter {
    /// Owns the constant-key thought lifecycle — the ends this wire never
    /// announces are derived by the shared lifecycle, not hand-rolled here.
    /// All accumulation lives in the shared accumulator.
    reasoning: crate::providers::internal::chunk_lifecycle::MintedReasoningLifecycle,
    /// A provider `error` event ended the turn; later frames are dead — the
    /// provider aborted, and interpreting more output (or a terminal) would
    /// dress the failure up as a completed turn.
    failed: bool,
    /// Function-call steps whose arguments may still stream as
    /// `arguments_delta` fragments: the shared index → grammar-identity
    /// bridge, keyed by the wire's step index. The wire announces the call
    /// in `step.start` (usually with `"arguments": {}`, kept as the slot's
    /// replace-if-no-deltas fallback), fragments the real payload across
    /// `step.delta` `arguments_delta` events, and closes it with
    /// `step.stop` — a genuine start/delta/end lifecycle. Recorded live in
    /// `streaming_grammar/interactions_same_tool_twice`; the pre-fix code
    /// emitted the empty-args call at `step.start` and dropped every
    /// fragment.
    ///
    /// The bridge's minter is also the whole-call minter
    /// ([`ToolCallBridge::minted_ids`]): both id-less paths draw from ONE
    /// counter, so a step assembly and a whole call can never collide on
    /// one minted key (the step-0 assembly used to share
    /// `Minted(Tool, 0)` with every id-less whole call, and the whole call
    /// silently swallowed the open assembly).
    open_function_steps: ToolCallBridge<u32>,
}

impl Default for InteractionsAdapter {
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

impl WireAdapter for InteractionsAdapter {
    type Frame = WireFrame;
    type Event = InteractionSseEvent;

    fn classify(&self, frame: WireFrame) -> WireEvent<InteractionSseEvent> {
        classify_interaction_frame(&frame.as_str())
    }

    fn interpret(&mut self, event: InteractionSseEvent, out: &mut AdapterOutput) {
        if self.failed {
            return;
        }

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
                                tool_events: Vec::new(),
                            },
                            out,
                        );
                    }
                }
                ContentDelta::ThoughtSignature(ThoughtSignatureDelta { signature }) => {
                    // One lifecycle end covers every shape (open block,
                    // already-closed block, signature-only stream); the
                    // shared accumulator signs the right part — the missing
                    // empty-buffer branch class (84a43e9e #2) cannot recur
                    // because there is no branch.
                    self.reasoning.emit_chunk(
                        ChunkParts {
                            reasoning: None,
                            reasoning_signature: Some(signature),
                            text: None,
                            tool_events: Vec::new(),
                        },
                        out,
                    );
                }
                delta => {
                    if let Some(parts) =
                        content_delta_to_parts(delta, self.open_function_steps.minted_ids())
                    {
                        // Interleaving content ends an open thought block —
                        // the shared lifecycle synthesizes the boundary end.
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
                    // A function-call step opens an ASSEMBLY: the wire may
                    // fragment the arguments as later `arguments_delta`
                    // events at this index, so emitting a whole call here
                    // would freeze the (usually empty) start-event payload
                    // and drop every fragment. The bridge keys by the
                    // wire's own id when present (never the tool name),
                    // minting from the shared counter otherwise.
                    let slot = self
                        .open_function_steps
                        .open(index, id.as_deref(), Some(&name));
                    // The announce payload is NOT a fragment: fragments
                    // append, and an announce that carries a partial (or
                    // full) payload alongside later `arguments_delta`
                    // events would concatenate into `{..}{..}`. It is kept
                    // as the slot's fallback, used only when no fragment
                    // ever arrives (replace-if-no-deltas).
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
                            tool_events,
                        },
                        out,
                    );
                } else {
                    // Every convertible item in wire order, each declared as
                    // its own chunk: the first one interleaving an open
                    // thought block ends it (the shared lifecycle synthesizes
                    // the boundary end once), and text lands in the active
                    // text block between the calls exactly where the wire
                    // put it.
                    for parts in step_start_to_parts(step, self.open_function_steps.minted_ids()) {
                        self.reasoning.emit_chunk(parts, out);
                    }
                }
            }
            InteractionSseEvent::StepStop { index, .. } => {
                // The wire promised a complete function-call step: close its
                // assembly. Malformed accumulated input surfaces in-band
                // (`Error` policy), matching the other complete-block wires.
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
                if let Some(usage) = interaction.usage.as_ref() {
                    span.record_token_usage(&crate::completion::Usage::from(usage));
                }

                // A function-call step still open here was announced by
                // `step.start` and — per this very event — belongs to a turn
                // the provider COMPLETED: its `step.stop` was lost or
                // reordered, not truncated away. Close each assembly with a
                // synthesized end so the announced call finalizes from its
                // accumulated fragments instead of vanishing in the
                // accumulator's end-of-stream clear (which is reserved for
                // genuine truncation, where the turn never finished). Wire
                // (announcement) order keeps parallel calls deterministic.
                for (index, slot) in self.open_function_steps.drain_ordered_indexed() {
                    tracing::debug!(
                        index,
                        "closing a function-call step left open at interaction.completed"
                    );
                    out.push(Ok(function_step_end(&slot)));
                }

                // Only a genuine `interaction.completed` event counts as the
                // provider completing the turn; the driver stops consuming
                // after the terminal record. EOF without one is truncation and
                // synthesizes nothing (see `finish`).
                //
                // The finish reason comes from the completed interaction's
                // lifecycle status — the API has no `finishReason` field —
                // and is absent when the interaction carries none.
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
                    streaming::StreamFinal::new(PROVIDER_NAME, usage)
                        .with_optional_finish_reason(finish_reason)
                        .with_optional_response_id(message_id)
                        .with_optional_model(native.model_version.as_deref())
                        .with_raw(raw),
                );
            }
            event @ InteractionSseEvent::Error { .. } => {
                // Preserve the provider error payload (code + message) as the
                // error body, matching the blocking path's
                // `completion_error_from_body`. The event is re-serialized
                // from its decoded form — the modeled fields survive. The
                // error arrives over an established stream, so there is no
                // HTTP status to attach (status: None).
                self.failed = true;
                let body = serde_json::to_string(&event).unwrap_or_default();
                out.push(Err(crate::provider_response::completion_error_from_body(
                    body,
                )));
            }
            InteractionSseEvent::InteractionCreated { .. }
            | InteractionSseEvent::InteractionStatusUpdate { .. } => {}
        }
    }

    fn finish(&mut self, _out: &mut AdapterOutput) {
        // EOF without `interaction.completed` is truncation: no terminal
        // record may be synthesized — it would report a successful completion
        // for a turn the provider aborted.
    }

    fn is_finished(&self) -> bool {
        // A provider `error` event is the wire's own in-band terminal:
        // `interpret` already pushed the `Err` and gates itself on `failed`,
        // so the driver must stop reading rather than drain the rest of the
        // transport (and pass through post-error unknown frames).
        self.failed
    }
}

pub(crate) fn stream_interaction_events<T>(
    client: super::InteractionsClient<T>,
    request: Request<Vec<u8>>,
) -> InteractionEventStream
where
    T: HttpClientExt + Clone + 'static,
{
    let mut event_source = GenericEventSource::new(client, request);

    let stream = stream! {
        while let Some(event_result) = event_source.next().await {
            match event_result {
                Ok(Event::Open) => continue,
                Ok(Event::Message(message)) => {
                    if message.data.trim().is_empty() {
                        continue;
                    }

                    // Same frame-triage table as the completion path's
                    // `run_wire_stream` driver — this surface yields typed
                    // events rather than grammar events, so it applies the
                    // driver's factored per-frame policy against the same
                    // classify site instead of restating the table.
                    match triage_frame(classify_interaction_frame(&message.data)) {
                        Ok(TriagedFrame::Event(event)) => yield Ok(event),
                        // This surface yields typed interaction events, not
                        // grammar events — there is no raw passthrough item to
                        // carry an unknown frame on, so it stays a warned skip
                        // (the completion path surfaces Unknown via the
                        // driver's `StreamEvent::Unknown` passthrough).
                        Ok(TriagedFrame::Unknown(_)) => {}
                        Err(error) => yield Err(error),
                    }
                }
                Err(crate::http_client::Error::StreamEnded) => break,
                Err(error) => {
                    tracing::error!(?error, "SSE error");
                    yield Err(CompletionError::from_stream_transport(error));
                    break;
                }
            }
        }

        event_source.close();
    };

    Box::pin(stream)
}

/// Close an announced function-call step. The shared accumulator finalizes
/// the call from its accumulated fragments; a step that fragmented nothing
/// falls back to the payload it announced at `step.start` (and to a
/// parameterless `{}` when it announced none) — the slot's
/// replace-if-no-deltas fallback.
///
/// Interactions is a single-identifier wire: its id travels as `tool_id`
/// only (`ToolCallSlot::end_event`'s shape). Filling `call_id` too made
/// the accumulator take the dual-wire arm and store
/// ProviderCallId{item_id: Some(fc_…)} — a fabricated Responses-shaped
/// identity that slips past the foreign-id guard on cross-provider replay.
/// Malformed accumulated input surfaces in-band (`Error` policy), matching
/// the other complete-block wires.
fn function_step_end(
    slot: &crate::providers::internal::tool_call_bridge::ToolCallSlot,
) -> streaming::StreamEvent {
    slot.end_event(streaming::UnparseableToolInput::Error)
}

/// A whole function call as one declared chunk (its start and end in the
/// tool-event slot).
fn function_call_parts(
    name: String,
    arguments: Option<Value>,
    id: Option<String>,
    tool_ids: &mut streaming::SyntheticIds,
) -> ChunkParts {
    // The wire's id when present; never the tool name — a name-as-id
    // fallback collides two same-tool calls in one turn.
    ChunkParts {
        reasoning: None,
        reasoning_signature: None,
        text: None,
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
        tool_events: Vec::new(),
    }
}

fn step_start_to_parts(step: Step, tool_ids: &mut streaming::SyntheticIds) -> Vec<ChunkParts> {
    match step {
        // Every convertible item, in wire order: a `model_output` step can
        // interleave text and function calls in one `content` list, and
        // keeping only the first silently dropped the rest.
        Step::ModelOutput { content } => content
            .into_iter()
            .filter_map(|content| content_to_parts(content, tool_ids))
            .collect(),
        Step::FunctionCall(FunctionCallContent {
            name,
            arguments,
            id,
        }) => {
            let Some(name) = name else {
                return Vec::new();
            };
            vec![function_call_parts(name, arguments, id, tool_ids)]
        }
        _ => Vec::new(),
    }
}

fn content_to_parts(
    content: Content,
    tool_ids: &mut streaming::SyntheticIds,
) -> Option<ChunkParts> {
    match content {
        Content::Text(text) if !text.text.is_empty() => Some(text_parts(text.text)),
        Content::FunctionCall(content) => {
            step_start_to_parts(Step::FunctionCall(content), tool_ids)
                .into_iter()
                .next()
        }
        _ => None,
    }
}

fn content_delta_to_parts(
    delta: ContentDelta,
    tool_ids: &mut streaming::SyntheticIds,
) -> Option<ChunkParts> {
    match delta {
        ContentDelta::Text(TextDelta {
            text: Some(text), ..
        }) => Some(text_parts(text)),
        ContentDelta::FunctionCall(FunctionCallContent {
            name,
            arguments,
            id,
        }) => {
            let name = name?;
            Some(function_call_parts(name, arguments, id, tool_ids))
        }
        // Thought deltas (`thought_summary`, `thought_signature`) are
        // stateful — the adapter accumulates and restates them in
        // `interpret`, so they never reach this stateless mapping.
        _ => None,
    }
}

#[cfg(test)]
mod tests;
