use crate::types::assistant_content::{PROVIDER_NAME, map_stop_reason, normalize_usage};
use crate::types::completion_request::AwsCompletionRequest;
use crate::types::converse_output::{StopReason, TokenUsage};
use crate::{
    completion::{CompletionModel, resolve_request_model},
    types::errors::{AwsSdkConverseStreamError, converse_stream_output_completion_error},
};
use async_stream::stream;
use aws_sdk_bedrockruntime::types as aws_bedrock;
use base64::{Engine, prelude::BASE64_STANDARD};
use rig_core::providers::internal::adapter::{AdapterOutput, WireAdapter, run_wire_stream};
use rig_core::providers::internal::tool_call_bridge::ToolCallBridge;
use rig_core::providers::internal::wire::{self, TypedEvent, WireEvent};
use rig_core::streaming::StreamingCompletionResponse;
use rig_core::telemetry::{CompletionOperation, CompletionSpanBuilder, SpanCombinator};
use rig_core::{
    completion::CompletionError,
    message::ReasoningContent,
    streaming::{RawStreamingChoice, ToolCallDeltaContent, UnparseableToolInput},
    wasm_compat::WasmCompatSend,
};
use serde::{Deserialize, Serialize};
use tracing_futures::Instrument;

#[derive(Clone, Deserialize, Serialize)]
pub struct BedrockStreamingResponse {
    pub usage: Option<TokenUsage>,
    /// Bedrock's own `stopReason` from the terminal `MessageStop` event, when
    /// the stream reported one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<StopReason>,
    /// The AWS request id from the converse-stream response's metadata
    /// (`x-amzn-RequestId`) — not part of any stream event; stamped by
    /// `raw_stream` from the SDK operation output, matching the unary
    /// surface's semantics. `None` when the SDK reported none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_request_id: Option<String>,
}

impl From<&BedrockStreamingResponse> for rig_core::completion::Usage {
    fn from(response: &BedrockStreamingResponse) -> Self {
        response
            .usage
            .as_ref()
            .map(normalize_usage)
            .unwrap_or_default()
    }
}

#[derive(Default)]
struct ReasoningState {
    /// Signature carried by this block's `signature` delta — the only
    /// adapter-side state, because the wire delivers it out of band from the
    /// thinking text. Thinking TEXT accumulates in the shared accumulator via
    /// `ReasoningDelta`s; no restatement buffer exists.
    signature: Option<String>,
}

/// Minted block identity for a Converse `contentBlockIndex`.
///
/// The wire index is `i32`; the minted index space is unsigned. The offset
/// map is injective over the whole `i32` domain, so even a (spec-violating)
/// negative index yields a distinct, well-formed minted identity instead of
/// a rendering the identity machinery could disagree about — the mint stays
/// total instead of trusting the wire.
fn block_id(content_block_index: i32) -> rig_core::streaming::StreamPartId {
    let index = (i64::from(content_block_index) - i64::from(i32::MIN)) as u64;
    rig_core::streaming::MintKind::Block.for_wire_index(index)
}

/// Close the open thinking block for `content_block_index`.
///
/// The end carries no restatement — the shared accumulator already holds every
/// `ReasoningDelta` this block streamed, so restating the text would supersede
/// the accumulation with a second copy of itself — only the signature, which
/// the wire never restates. Adaptive-thinking blocks can even be
/// signature-only (a `Signature` delta with no non-empty `Text` delta), and
/// dropping that signature makes the next turn fail with
/// `messages.N.content.0.thinking.signature: Field required` on replay. A
/// wholly empty block still lands nowhere: a payload-less end creates no part.
fn reasoning_end(
    state: ReasoningState,
    content_block_index: i32,
) -> RawStreamingChoice<BedrockStreamingResponse> {
    RawStreamingChoice::ReasoningEnd {
        // Bedrock has no reasoning item id; the block's `contentBlockIndex`
        // is stable across its deltas and its close.
        id: block_id(content_block_index),
        reasoning: None,
        signature: state.signature,
        // Both call sites close on a frame the wire actually sent — its own
        // `contentBlockStop`, or the redacted sibling delta that ends the
        // plaintext block — so the completed block reaches the consumer.
        wire_sent: true,
    }
}

/// Accumulated per-stream state for [`process_event`].
///
/// In-flight tool calls are keyed by Bedrock's own `contentBlockIndex` via
/// the shared [`ToolCallBridge`]: the Converse stream indexes every content
/// block, and a message may open several tool-use blocks before it stops, so
/// a single "current" slot would let a later block silently overwrite an
/// earlier one. Fragment assembly, internal-id minting, and finalize policy
/// live in the shared accumulator (`PartsAccumulator::tool_input_*`); the
/// bridge keeps only the index → identity mapping (and the name, for the
/// dropped-block warning).
#[derive(Default)]
struct StreamState {
    tool_calls: ToolCallBridge<i32>,
    current_reasoning: Option<ReasoningState>,
    final_stop_reason: Option<StopReason>,
}

/// A static, log-safe label for a stop reason: known variants map to their
/// wire spelling, `Unknown` collapses to `"other"` so its carried wire
/// string (potentially model output) never reaches a log line.
fn stop_reason_label(stop_reason: &StopReason) -> &'static str {
    match stop_reason {
        StopReason::ContentFiltered => "content_filtered",
        StopReason::EndTurn => "end_turn",
        StopReason::GuardrailIntervened => "guardrail_intervened",
        StopReason::MaxTokens => "max_tokens",
        StopReason::StopSequence => "stop_sequence",
        StopReason::ToolUse => "tool_use",
        StopReason::Unknown(_) => "other",
    }
}

/// Handle one Converse stream event, returning the items to yield in order.
///
/// Kept as a plain function over [`StreamState`] so the event bookkeeping can
/// be unit-tested without an AWS event receiver.
fn process_event(
    state: &mut StreamState,
    output: aws_bedrock::ConverseStreamOutput,
) -> Vec<Result<RawStreamingChoice<BedrockStreamingResponse>, CompletionError>> {
    let mut items = Vec::new();
    match output {
        aws_bedrock::ConverseStreamOutput::ContentBlockDelta(event) => {
            let Some(delta) = event.delta else {
                tracing::warn!("skipping ContentBlockDelta with a missing delta");
                return items;
            };
            match delta {
                aws_bedrock::ContentBlockDelta::Text(text) => {
                    items.push(Ok(RawStreamingChoice::Message(text)));
                }
                aws_bedrock::ContentBlockDelta::ToolUse(tool) => {
                    if let Some(tool_call) = state.tool_calls.get(event.content_block_index) {
                        // Emit the delta so UI can show progress; the shared
                        // accumulator assembles the fragments.
                        items.push(Ok(RawStreamingChoice::ToolCallDelta {
                            id: tool_call.key().to_owned(),
                            content: ToolCallDeltaContent::Delta(tool.input().to_string()),
                        }));
                    }
                }
                aws_bedrock::ContentBlockDelta::ReasoningContent(reasoning) => match reasoning {
                    aws_bedrock::ReasoningContentBlockDelta::Text(text) => {
                        // Marks the block open so its stop emits an end; the
                        // text itself belongs to the shared accumulator.
                        state
                            .current_reasoning
                            .get_or_insert_with(ReasoningState::default);

                        if !text.is_empty() {
                            items.push(Ok(RawStreamingChoice::ReasoningDelta {
                                reasoning: text.clone(),
                                // Derive identity from `contentBlockIndex`
                                // (no wire id on Converse reasoning blocks).
                                id: block_id(event.content_block_index),
                                provider_id: None,
                            }));
                        }
                    }
                    aws_bedrock::ReasoningContentBlockDelta::Signature(signature) => {
                        state
                            .current_reasoning
                            .get_or_insert_with(ReasoningState::default)
                            .signature = Some(signature.clone());
                    }
                    aws_bedrock::ReasoningContentBlockDelta::RedactedContent(blob) => {
                        // Opaque provider state the safety classifier
                        // encrypted. It is not part of any plaintext thinking
                        // block, so close an open one first: sharing the
                        // block index would otherwise make the redacted block
                        // *replace* the delta-built thinking part instead of
                        // landing beside it as a sibling.
                        if let Some(open) = state.current_reasoning.take() {
                            items.push(Ok(reasoning_end(open, event.content_block_index)));
                        }

                        items.push(Ok(RawStreamingChoice::Reasoning {
                            // Same minted block identity as the sibling
                            // reasoning paths, so provenance and boundary
                            // semantics stay uniform.
                            id: block_id(event.content_block_index),
                            provider_id: None,
                            content: ReasoningContent::Redacted {
                                // The wire carries raw bytes; rig's canonical
                                // reasoning content is a string, so the blob
                                // travels base64-encoded and decodes back on
                                // the way out.
                                data: BASE64_STANDARD.encode(blob.as_ref()),
                            },
                        }));
                    }
                    unknown => {
                        tracing::warn!(
                            delta = ?std::mem::discriminant(&unknown),
                            "skipping unrecognized Bedrock reasoning content delta variant"
                        );
                    }
                },
                unknown => {
                    tracing::warn!(
                        delta = ?std::mem::discriminant(&unknown),
                        "skipping unrecognized Bedrock content block delta variant"
                    );
                }
            }
        }
        aws_bedrock::ConverseStreamOutput::ContentBlockStart(event) => {
            let Some(start) = event.start else {
                tracing::warn!("skipping ContentBlockStart with no data");
                return items;
            };
            match start {
                aws_bedrock::ContentBlockStart::ToolUse(tool_use) => {
                    // The wire always supplies a tool-use id here; the shared
                    // bridge fixes it as the assembly key (and would mint one
                    // in the reserved namespace if the wire ever omitted it).
                    let slot = state.tool_calls.open(
                        event.content_block_index,
                        Some(&tool_use.tool_use_id),
                        Some(&tool_use.name),
                    );
                    items.push(Ok(RawStreamingChoice::ToolCallDelta {
                        id: slot.key().to_owned(),
                        content: ToolCallDeltaContent::Name(tool_use.name),
                    }));
                }
                // `ContentBlockStart` is a union: `toolUse` is the only
                // variant modeled today, and a future one is not a stream
                // failure. Failing the turn here contradicted every sibling
                // arm (which warn and skip) and the classify layer's
                // Unknown-frame policy, and the message ("Stream is empty")
                // described neither the frame nor the cause.
                unknown => tracing::warn!(
                    start = ?std::mem::discriminant(&unknown),
                    "skipping unrecognized Bedrock ContentBlockStart variant"
                ),
            }
        }
        aws_bedrock::ConverseStreamOutput::ContentBlockStop(event) => {
            if let Some(reasoning_state) = state.current_reasoning.take() {
                items.push(Ok(reasoning_end(
                    reasoning_state,
                    event.content_block_index,
                )));
            }
            // A closed tool-use block is complete: finalize and emit it here,
            // mirroring the reasoning close above, so every call in a
            // multi-tool-call message reaches the consumer. The shared
            // accumulator finalizes the assembled input: an empty accumulated
            // input means a tool with no parameters, and malformed JSON
            // surfaces as an error item (`UnparseableToolInput::Error`) rather
            // than a silent drop, so the terminal can never report tool use
            // whose calls the consumer never saw.
            if let Some(tool_call) = state.tool_calls.remove(event.content_block_index) {
                items.push(Ok(RawStreamingChoice::ToolInputEnd(
                    tool_call.end_event(UnparseableToolInput::Error),
                )));
            }
        }
        aws_bedrock::ConverseStreamOutput::MessageStop(message_stop_event) => {
            // Remember Bedrock's own terminal reason so the final
            // record can report it; an unmapped SDK variant is kept
            // verbatim rather than dropped.
            state.final_stop_reason = Some(
                StopReason::try_from(message_stop_event.stop_reason.clone()).unwrap_or_else(|_| {
                    StopReason::Unknown(crate::types::converse_output::UnknownVariantValue(
                        message_stop_event.stop_reason.as_str().to_owned(),
                    ))
                }),
            );
            // Tool calls normally flush at their ContentBlockStop; when the
            // message genuinely stopped for tool use, drain any stragglers
            // here defensively (in block order) so a stream that omits the
            // stop event still delivers every call. Under any other stop
            // reason (notably MaxTokens) an in-flight block is one the model
            // never finished: drop it rather than fabricate a `{}`-args call
            // or a spurious error item — truncation is signaled to the
            // consumer by the mapped finish reason on the terminal record,
            // which the Metadata path emits via `final_stop_reason`.
            if matches!(state.final_stop_reason, Some(StopReason::ToolUse)) {
                for tool_call in state.tool_calls.drain_ordered() {
                    items.push(Ok(RawStreamingChoice::ToolInputEnd(
                        tool_call.end_event(UnparseableToolInput::Error),
                    )));
                }
            } else if !state.tool_calls.is_empty() {
                // Structural metadata only: tool names can be model-chosen
                // (a hallucinated call's name is model output) and the
                // `Unknown` variant carries a wire string, so neither may
                // reach the WARN log. Known variants log a static label —
                // unknown ones collapse to "other", never the wire value.
                let dropped = state.tool_calls.drain_ordered().len();
                tracing::warn!(
                    dropped_tool_calls = dropped,
                    stop_reason = state
                        .final_stop_reason
                        .as_ref()
                        .map_or("none", stop_reason_label),
                    "dropping unfinished tool-use blocks left in flight at MessageStop"
                );
            }
        }
        aws_bedrock::ConverseStreamOutput::Metadata(metadata_event) => {
            // Extract usage information from metadata; a missing usage still
            // yields a terminal record so the stream ends with a FinalResponse.
            let final_response = BedrockStreamingResponse {
                // The mirror conversion is infallible for `TokenUsage`.
                usage: metadata_event
                    .usage
                    .and_then(|usage| TokenUsage::try_from(usage).ok()),
                stop_reason: state.final_stop_reason.clone(),
                // Stamped by `raw_stream`; the adapter never sees the SDK
                // operation output's metadata.
                provider_request_id: None,
            };
            items.push(Ok(RawStreamingChoice::FinalResponse(final_response)));
        }
        _ => {}
    }
    items
}

impl WireAdapter for StreamState {
    type Frame = aws_bedrock::ConverseStreamOutput;
    type Event = aws_bedrock::ConverseStreamOutput;
    type Response = BedrockStreamingResponse;

    fn classify(&self, frame: Self::Frame) -> WireEvent<Self::Event> {
        // The AWS SDK already deserialized the event-stream frame, so the
        // byte-level decode step collapses: an event-stream decode failure
        // surfaces as a receive error on the transport, and the only triage
        // left here is the SDK's own unknown-variant signal on its
        // non-exhaustive union.
        wire::classify_typed_event(if frame.is_unknown() {
            TypedEvent::Unrecognized {
                event_type: "unknown".to_string(),
                detail: format!("{frame:?}"),
            }
        } else {
            TypedEvent::Modeled(frame)
        })
    }

    fn interpret(&mut self, event: Self::Event, out: &mut AdapterOutput<Self::Response>) {
        for item in process_event(self, event) {
            if let Ok(RawStreamingChoice::FinalResponse(final_response)) = &item {
                tracing::Span::current().record_token_usage(&final_response.into());
            }
            out.push(item);
        }
    }

    fn finish(&mut self, _out: &mut AdapterOutput<Self::Response>) {
        // EOF without Bedrock's `Metadata` terminal is truncation: in-flight
        // blocks drop and no terminal record may be synthesized.
    }
}

/// Drive already-typed Converse stream events through the full shared
/// pipeline — driver policy, canonical grammar, terminal normalization.
///
/// The events-first conformance seam: the adapter is a pure
/// `(state, event) → events` function, so grammar scenarios feed SDK events
/// directly with no AWS transport.
pub fn stream_from_events(
    events: impl futures::Stream<Item = Result<aws_bedrock::ConverseStreamOutput, CompletionError>>
    + WasmCompatSend
    + 'static,
) -> StreamingCompletionResponse {
    let raw = run_wire_stream(events, StreamState::default());
    StreamingCompletionResponse::stream(PROVIDER_NAME, normalize_bedrock_stream(raw))
}

fn normalize_bedrock_stream(
    raw: rig_core::streaming::RawStreamingResult<BedrockStreamingResponse>,
) -> rig_core::streaming::StreamingResult {
    rig_core::streaming::normalize_stream(raw, |response| {
        let usage = (&response).into();
        let finish_reason = response.stop_reason.as_ref().map(map_stop_reason);
        Ok(rig_core::streaming::StreamFinal::new(PROVIDER_NAME, usage)
            .with_optional_provider_request_id(response.provider_request_id.clone())
            .with_optional_finish_reason(finish_reason))
    })
}

impl CompletionModel {
    /// Open a stream whose terminal record stays Bedrock's own response type.
    pub async fn raw_stream(
        &self,
        completion_request: rig_core::completion::CompletionRequest,
    ) -> Result<rig_core::streaming::RawStreamingResult<BedrockStreamingResponse>, CompletionError>
    {
        let request_model = resolve_request_model(&self.model, &completion_request);
        let system_instructions = completion_request.preamble.clone();
        let record_telemetry_content = completion_request.record_telemetry_content;
        let request = AwsCompletionRequest {
            inner: completion_request,
            prompt_caching: self.prompt_caching,
        };
        let span = CompletionSpanBuilder::new(
            "aws_bedrock",
            &request_model,
            CompletionOperation::ChatStreaming,
        )
        .system_instructions(system_instructions.as_deref(), record_telemetry_content)
        .build();

        let mut converse_builder = self
            .client
            .get_inner()
            .await
            .converse_stream()
            .model_id(request_model);

        let tool_config = request.tools_config()?;
        let prompt_with_history = request.messages()?;
        let output_config = request.output_config()?;
        converse_builder = converse_builder
            .set_additional_model_request_fields(request.additional_params())
            .set_inference_config(request.inference_config())
            .set_tool_config(tool_config)
            .set_system(request.system_prompt()?)
            .set_messages(Some(prompt_with_history))
            .set_output_config(output_config);

        let response = converse_builder
            .send()
            .instrument(span.clone())
            .await
            .map_err(|sdk_error| {
                Into::<CompletionError>::into(AwsSdkConverseStreamError(sdk_error))
            })?;

        // Read the AWS request id off the operation output *before* the event
        // stream is moved — `ConverseStreamOutput` implements the SDK
        // `RequestId` trait on the whole output, not on stream events.
        let provider_request_id =
            aws_sdk_bedrockruntime::operation::RequestId::request_id(&response).map(str::to_string);

        // Transport layer: SDK event-stream frames only — an event-stream
        // decode/receive failure is a transport error; classification and
        // policy live in the shared driver.
        let transport = stream! {
            let mut stream = response.stream;
            loop {
                match stream.recv().await {
                    Ok(Some(output)) => yield Ok(output),
                    Ok(None) => break,
                    Err(err) => {
                        yield Err(converse_stream_output_completion_error(err.into_service_error()));
                        break;
                    }
                }
            }
        };

        // Stamp the terminal record with the id captured above, mirroring the
        // unary surface (`InternalConverseOutput::request_id`).
        use futures::StreamExt as _;
        let stream = run_wire_stream(transport, StreamState::default()).instrument(span);
        Ok(Box::pin(stream.map(move |item| {
            item.map(|choice| match choice {
                RawStreamingChoice::FinalResponse(mut response) => {
                    response.provider_request_id = provider_request_id.clone();
                    RawStreamingChoice::FinalResponse(response)
                }
                other => other,
            })
        })))
    }

    /// Open a stream normalized to rig's terminal record. Delegates to
    /// [`CompletionModel::raw_stream`] — one request either way.
    pub(crate) async fn stream(
        &self,
        completion_request: rig_core::completion::CompletionRequest,
    ) -> Result<StreamingCompletionResponse, CompletionError> {
        let raw = self.raw_stream(completion_request).await?;

        Ok(StreamingCompletionResponse::stream(
            PROVIDER_NAME,
            normalize_bedrock_stream(raw),
        ))
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use rig_core::message::Reasoning;
    use rig_core::streaming::StreamedAssistantContent;

    // ---- Event-seam helpers: no AWS transport, `stream_from_events` only ----

    fn reasoning_text_delta(index: i32, text: &str) -> aws_bedrock::ConverseStreamOutput {
        aws_bedrock::ConverseStreamOutput::ContentBlockDelta(
            aws_bedrock::ContentBlockDeltaEvent::builder()
                .content_block_index(index)
                .delta(aws_bedrock::ContentBlockDelta::ReasoningContent(
                    aws_bedrock::ReasoningContentBlockDelta::Text(text.to_string()),
                ))
                .build()
                .expect("reasoning text delta should build"),
        )
    }

    fn reasoning_signature_delta(index: i32, signature: &str) -> aws_bedrock::ConverseStreamOutput {
        aws_bedrock::ConverseStreamOutput::ContentBlockDelta(
            aws_bedrock::ContentBlockDeltaEvent::builder()
                .content_block_index(index)
                .delta(aws_bedrock::ContentBlockDelta::ReasoningContent(
                    aws_bedrock::ReasoningContentBlockDelta::Signature(signature.to_string()),
                ))
                .build()
                .expect("reasoning signature delta should build"),
        )
    }

    fn reasoning_redacted_delta(index: i32, blob: &[u8]) -> aws_bedrock::ConverseStreamOutput {
        aws_bedrock::ConverseStreamOutput::ContentBlockDelta(
            aws_bedrock::ContentBlockDeltaEvent::builder()
                .content_block_index(index)
                .delta(aws_bedrock::ContentBlockDelta::ReasoningContent(
                    aws_bedrock::ReasoningContentBlockDelta::RedactedContent(
                        aws_smithy_types::Blob::new(blob.to_vec()),
                    ),
                ))
                .build()
                .expect("redacted reasoning delta should build"),
        )
    }

    fn block_stop(index: i32) -> aws_bedrock::ConverseStreamOutput {
        aws_bedrock::ConverseStreamOutput::ContentBlockStop(
            aws_bedrock::ContentBlockStopEvent::builder()
                .content_block_index(index)
                .build()
                .expect("content block stop should build"),
        )
    }

    fn terminal() -> Vec<aws_bedrock::ConverseStreamOutput> {
        vec![
            aws_bedrock::ConverseStreamOutput::MessageStop(
                aws_bedrock::MessageStopEvent::builder()
                    .stop_reason(aws_bedrock::StopReason::EndTurn)
                    .build()
                    .expect("message stop should build"),
            ),
            aws_bedrock::ConverseStreamOutput::Metadata(
                aws_bedrock::ConverseStreamMetadataEvent::builder().build(),
            ),
        ]
    }

    struct Drained {
        reasoning: Vec<Reasoning>,
        errors: Vec<String>,
        reached_terminal: bool,
    }

    async fn drain(events: Vec<aws_bedrock::ConverseStreamOutput>) -> Drained {
        let mut stream = stream_from_events(futures::stream::iter(events.into_iter().map(Ok)));
        let mut drained = Drained {
            reasoning: Vec::new(),
            errors: Vec::new(),
            reached_terminal: false,
        };

        while let Some(item) = stream.next().await {
            match item {
                Ok(StreamedAssistantContent::Reasoning { reasoning, .. }) => {
                    drained.reasoning.push(reasoning);
                }
                Ok(StreamedAssistantContent::Final(_)) => drained.reached_terminal = true,
                Ok(_) => {}
                Err(error) => drained.errors.push(error.to_string()),
            }
        }

        drained
    }

    /// Ordinary extended-thinking shape through the SHARED driver: thinking
    /// deltas, the block's whole-block close at `contentBlockStop`, then
    /// visible text. The driver's boundary law must treat the same-key whole
    /// block as a close — this exact stream used to abort every debug build
    /// (sequence-law O1).
    #[tokio::test]
    async fn thinking_then_text_streams_through_the_driver_without_violation() {
        let drained = drain(vec![
            reasoning_text_delta(0, "let me think"),
            block_stop(0),
            text_delta_event(1, "the answer"),
            block_stop_event(1),
            message_stop_event(aws_bedrock::StopReason::EndTurn),
        ])
        .await;
        assert!(drained.errors.is_empty(), "{:?}", drained.errors);
        assert_eq!(drained.reasoning.len(), 1);
        assert_eq!(
            drained
                .reasoning
                .iter()
                .flat_map(|reasoning| reasoning.content.iter())
                .cloned()
                .collect::<Vec<_>>(),
            vec![ReasoningContent::Text {
                text: "let me think".to_string(),
                signature: None,
            }],
            "an unsigned block closes carrying just its accumulated text"
        );
    }

    const REDACTED_BLOB: &[u8] = b"\x00opaque-stream-ciphertext\xff";

    /// #2258 F2(a): the redacted delta used to hit `_ => {}` and vanish.
    #[tokio::test]
    async fn redacted_reasoning_delta_reaches_the_consumer() {
        let mut events = vec![reasoning_redacted_delta(0, REDACTED_BLOB), block_stop(0)];
        events.extend(terminal());

        let drained = drain(events).await;

        assert!(drained.errors.is_empty(), "errors: {:?}", drained.errors);
        assert_eq!(
            drained
                .reasoning
                .iter()
                .flat_map(|reasoning| reasoning.content.iter())
                .cloned()
                .collect::<Vec<_>>(),
            vec![ReasoningContent::Redacted {
                data: BASE64_STANDARD.encode(REDACTED_BLOB),
            }]
        );
        assert!(drained.reached_terminal);
    }

    /// The redacted block must land BESIDE an open thinking block, not replace
    /// it: both share `block-{index}`, so without draining the open state
    /// first the accumulator would supersede the delta-built thinking part.
    #[tokio::test]
    async fn redacted_reasoning_is_a_sibling_of_the_open_thinking_block() {
        let mut events = vec![
            reasoning_text_delta(0, "visible thinking"),
            reasoning_signature_delta(0, "sig_1"),
            reasoning_redacted_delta(0, REDACTED_BLOB),
            block_stop(0),
        ];
        events.extend(terminal());

        let drained = drain(events).await;

        assert!(drained.errors.is_empty(), "errors: {:?}", drained.errors);
        let content: Vec<ReasoningContent> = drained
            .reasoning
            .iter()
            .flat_map(|reasoning| reasoning.content.iter())
            .cloned()
            .collect();
        assert_eq!(
            content,
            vec![
                ReasoningContent::Text {
                    text: "visible thinking".to_string(),
                    signature: Some("sig_1".to_string()),
                },
                ReasoningContent::Redacted {
                    data: BASE64_STANDARD.encode(REDACTED_BLOB),
                },
            ]
        );
        assert!(drained.reached_terminal);
    }

    /// #2258 H5: a non-`ToolUse` `ContentBlockStart` used to fail the whole
    /// stream with `ProviderError("Stream is empty")`.
    #[tokio::test]
    async fn non_tool_use_content_block_start_is_skipped_not_failed() {
        let mut events = vec![
            aws_bedrock::ConverseStreamOutput::ContentBlockStart(
                aws_bedrock::ContentBlockStartEvent::builder()
                    .content_block_index(0)
                    .start(aws_bedrock::ContentBlockStart::ToolResult(
                        aws_bedrock::ToolResultBlockStart::builder()
                            .tool_use_id("tool_1")
                            .build()
                            .expect("tool result start should build"),
                    ))
                    .build()
                    .expect("content block start should build"),
            ),
            block_stop(0),
        ];
        events.extend(terminal());

        let drained = drain(events).await;

        assert!(
            drained.errors.is_empty(),
            "an unmodeled ContentBlockStart must not fail the stream: {:?}",
            drained.errors
        );
        assert!(
            drained.reached_terminal,
            "the stream must still reach its terminal record"
        );
    }

    #[test]
    fn test_bedrock_usage_creation() {
        let usage = TokenUsage {
            input_tokens: 100,
            output_tokens: 50,
            total_tokens: 150,
            cache_read_input_tokens: None,
            cache_write_input_tokens: None,
        };

        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.total_tokens, 150);
    }

    #[test]
    fn test_bedrock_streaming_response_with_usage() {
        let response = BedrockStreamingResponse {
            usage: Some(TokenUsage {
                input_tokens: 200,
                output_tokens: 75,
                total_tokens: 275,
                cache_read_input_tokens: Some(40),
                cache_write_input_tokens: Some(10),
            }),
            stop_reason: None,
            provider_request_id: None,
        };

        assert_eq!(
            rig_core::completion::Usage::from(&response),
            rig_core::completion::Usage {
                input_tokens: 200,
                output_tokens: 75,
                total_tokens: 275,
                cached_input_tokens: 40,
                cache_creation_input_tokens: 10,
                tool_use_prompt_tokens: 0,
                reasoning_tokens: 0,
            }
        );
    }

    #[test]
    fn test_bedrock_streaming_response_without_usage() {
        let response = BedrockStreamingResponse {
            usage: None,
            stop_reason: None,
            provider_request_id: None,
        };

        // Zero-valued usage is rig's documented sentinel for "the provider
        // reported no usage metrics".
        assert_eq!(
            rig_core::completion::Usage::from(&response),
            rig_core::completion::Usage::new()
        );
        assert!(!rig_core::completion::Usage::from(&response).has_values());
    }

    #[test]
    fn test_streaming_response_normalizes_usage() {
        let response = BedrockStreamingResponse {
            usage: Some(TokenUsage {
                input_tokens: 448,
                output_tokens: 68,
                total_tokens: 516,
                cache_read_input_tokens: Some(80),
                cache_write_input_tokens: Some(20),
            }),
            stop_reason: None,
            provider_request_id: None,
        };

        // The streaming response normalizes into rig's usage record.
        assert_eq!(
            rig_core::completion::Usage::from(&response),
            rig_core::completion::Usage {
                input_tokens: 448,
                output_tokens: 68,
                total_tokens: 516,
                cached_input_tokens: 80,
                cache_creation_input_tokens: 20,
                tool_use_prompt_tokens: 0,
                reasoning_tokens: 0,
            }
        );
    }

    #[test]
    fn test_bedrock_usage_serde() {
        let usage = TokenUsage {
            input_tokens: 100,
            output_tokens: 50,
            total_tokens: 150,
            cache_read_input_tokens: Some(25),
            cache_write_input_tokens: Some(5),
        };

        // Test serialization
        let json = serde_json::to_string(&usage).expect("Should serialize");
        assert!(json.contains("\"input_tokens\":100"));
        assert!(json.contains("\"output_tokens\":50"));
        assert!(json.contains("\"total_tokens\":150"));

        // Test deserialization
        let deserialized: TokenUsage = serde_json::from_str(&json).expect("Should deserialize");
        assert_eq!(deserialized.input_tokens, usage.input_tokens);
        assert_eq!(deserialized.output_tokens, usage.output_tokens);
        assert_eq!(deserialized.total_tokens, usage.total_tokens);
        assert_eq!(
            deserialized.cache_read_input_tokens,
            usage.cache_read_input_tokens
        );
        assert_eq!(
            deserialized.cache_write_input_tokens,
            usage.cache_write_input_tokens
        );
    }

    #[test]
    fn test_bedrock_streaming_response_serde() {
        let response = BedrockStreamingResponse {
            usage: Some(TokenUsage {
                input_tokens: 200,
                output_tokens: 75,
                total_tokens: 275,
                cache_read_input_tokens: Some(30),
                cache_write_input_tokens: Some(15),
            }),
            stop_reason: None,
            provider_request_id: None,
        };

        // Test serialization
        let json = serde_json::to_string(&response).expect("Should serialize");
        assert!(json.contains("\"input_tokens\":200"));

        // Test deserialization
        let deserialized: BedrockStreamingResponse =
            serde_json::from_str(&json).expect("Should deserialize");
        assert!(deserialized.usage.is_some());
        let usage = deserialized.usage.unwrap();
        assert_eq!(usage.input_tokens, 200);
        assert_eq!(usage.output_tokens, 75);
        assert_eq!(usage.total_tokens, 275);
        assert_eq!(usage.cache_read_input_tokens, Some(30));
        assert_eq!(usage.cache_write_input_tokens, Some(15));
    }

    /// A signed thinking block closes with its signature attached to the
    /// text the shared accumulator assembled from the deltas — the exact
    /// shape the next turn must replay to Bedrock.
    #[tokio::test]
    async fn signed_thinking_block_closes_with_its_signature() {
        let mut events = vec![
            reasoning_text_delta(0, "I am "),
            reasoning_text_delta(0, "thinking"),
            reasoning_signature_delta(0, "sig-abc"),
            block_stop(0),
        ];
        events.extend(terminal());

        let drained = drain(events).await;

        assert!(drained.errors.is_empty(), "errors: {:?}", drained.errors);
        assert_eq!(
            drained
                .reasoning
                .iter()
                .flat_map(|reasoning| reasoning.content.iter())
                .cloned()
                .collect::<Vec<_>>(),
            vec![ReasoningContent::Text {
                text: "I am thinking".to_string(),
                signature: Some("sig-abc".to_string()),
            }]
        );
    }

    /// Adaptive thinking on Bedrock can produce a `Signature` delta with no
    /// non-empty `Text` delta. The signature is replay-required provider
    /// state, so a signature-only block must still reach the consumer —
    /// dropping it fails the next turn with
    /// `messages.N.content.0.thinking.signature: Field required`.
    #[tokio::test]
    async fn signature_only_thinking_block_still_reaches_the_consumer() {
        let mut events = vec![reasoning_signature_delta(0, "sig-only"), block_stop(0)];
        events.extend(terminal());

        let drained = drain(events).await;

        assert!(drained.errors.is_empty(), "errors: {:?}", drained.errors);
        assert_eq!(
            drained
                .reasoning
                .iter()
                .flat_map(|reasoning| reasoning.content.iter())
                .cloned()
                .collect::<Vec<_>>(),
            vec![ReasoningContent::Text {
                text: String::new(),
                signature: Some("sig-only".to_string()),
            }]
        );
    }

    /// A block that streamed nothing at all — an empty `Text` delta and no
    /// signature — says nothing at its stop: the payload-less end must not
    /// conjure an empty reasoning part.
    #[tokio::test]
    async fn wholly_empty_thinking_block_emits_nothing() {
        let mut events = vec![reasoning_text_delta(0, ""), block_stop(0)];
        events.extend(terminal());

        let drained = drain(events).await;

        assert!(drained.errors.is_empty(), "errors: {:?}", drained.errors);
        assert!(drained.reasoning.is_empty());
        assert!(drained.reached_terminal);
    }

    fn tool_start_event(index: i32, id: &str, name: &str) -> aws_bedrock::ConverseStreamOutput {
        aws_bedrock::ConverseStreamOutput::ContentBlockStart(
            aws_bedrock::ContentBlockStartEvent::builder()
                .content_block_index(index)
                .start(aws_bedrock::ContentBlockStart::ToolUse(
                    aws_bedrock::ToolUseBlockStart::builder()
                        .tool_use_id(id)
                        .name(name)
                        .build()
                        .expect("tool use start should build"),
                ))
                .build()
                .expect("content block start should build"),
        )
    }

    fn tool_delta_event(index: i32, input: &str) -> aws_bedrock::ConverseStreamOutput {
        aws_bedrock::ConverseStreamOutput::ContentBlockDelta(
            aws_bedrock::ContentBlockDeltaEvent::builder()
                .content_block_index(index)
                .delta(aws_bedrock::ContentBlockDelta::ToolUse(
                    aws_bedrock::ToolUseBlockDelta::builder()
                        .input(input)
                        .build()
                        .expect("tool use delta should build"),
                ))
                .build()
                .expect("content block delta should build"),
        )
    }

    fn text_delta_event(index: i32, text: &str) -> aws_bedrock::ConverseStreamOutput {
        aws_bedrock::ConverseStreamOutput::ContentBlockDelta(
            aws_bedrock::ContentBlockDeltaEvent::builder()
                .content_block_index(index)
                .delta(aws_bedrock::ContentBlockDelta::Text(text.to_string()))
                .build()
                .expect("content block delta should build"),
        )
    }

    fn block_stop_event(index: i32) -> aws_bedrock::ConverseStreamOutput {
        aws_bedrock::ConverseStreamOutput::ContentBlockStop(
            aws_bedrock::ContentBlockStopEvent::builder()
                .content_block_index(index)
                .build()
                .expect("content block stop should build"),
        )
    }

    fn message_stop_event(reason: aws_bedrock::StopReason) -> aws_bedrock::ConverseStreamOutput {
        aws_bedrock::ConverseStreamOutput::MessageStop(
            aws_bedrock::MessageStopEvent::builder()
                .stop_reason(reason)
                .build()
                .expect("message stop should build"),
        )
    }

    /// Run a sequence of events through [`process_event`] with fresh state,
    /// returning every item the stream would yield, plus the final state.
    fn run_events(
        events: Vec<aws_bedrock::ConverseStreamOutput>,
    ) -> (
        Vec<Result<RawStreamingChoice<BedrockStreamingResponse>, CompletionError>>,
        StreamState,
    ) {
        let mut state = StreamState::default();
        let mut items = Vec::new();
        for event in events {
            items.extend(process_event(&mut state, event));
        }
        (items, state)
    }

    /// Drive the raw items through the same normalized pipeline the public
    /// stream uses (terminal mapping plus the shared accumulator), returning
    /// the completed tool calls and the in-band errors a consumer would see.
    /// Tool-call finalization happens in the accumulator, so assertions about
    /// completed calls and malformed-input errors belong at this level.
    async fn assembled(
        items: Vec<Result<RawStreamingChoice<BedrockStreamingResponse>, CompletionError>>,
    ) -> (Vec<rig_core::message::ToolCall>, Vec<CompletionError>) {
        use futures::StreamExt;
        let raw: rig_core::streaming::RawStreamingResult<BedrockStreamingResponse> =
            Box::pin(futures::stream::iter(items));
        let mut stream =
            StreamingCompletionResponse::stream(PROVIDER_NAME, normalize_bedrock_stream(raw));
        let mut calls = Vec::new();
        let mut errors = Vec::new();
        while let Some(item) = stream.next().await {
            match item {
                Ok(rig_core::streaming::StreamedAssistantContent::ToolCall {
                    tool_call, ..
                }) => calls.push(tool_call),
                Err(err) => errors.push(err),
                Ok(_) => {}
            }
        }
        (calls, errors)
    }

    #[tokio::test]
    async fn parallel_tool_calls_all_emitted_with_tool_use_terminal() {
        // Two tool-use blocks in one message: both must survive, and the
        // latched stop reason must map to a tool-use terminal.
        let (items, state) = run_events(vec![
            tool_start_event(0, "call_a", "get_weather"),
            tool_delta_event(0, "{\"location\":"),
            tool_delta_event(0, "\"Paris\"}"),
            block_stop_event(0),
            tool_start_event(1, "call_b", "get_time"),
            tool_delta_event(1, "{\"zone\":\"UTC\"}"),
            block_stop_event(1),
            message_stop_event(aws_bedrock::StopReason::ToolUse),
        ]);

        assert!(items.iter().all(|item| item.is_ok()));
        // The terminal reports tool use with the calls actually delivered.
        assert_eq!(state.final_stop_reason, Some(StopReason::ToolUse));
        assert_eq!(
            map_stop_reason(&StopReason::ToolUse),
            rig_core::completion::FinishReason::ToolCalls
        );

        let (calls, errors) = assembled(items).await;
        assert!(errors.is_empty());
        assert_eq!(calls.len(), 2, "both parallel tool calls must be emitted");
        let first = calls.first().expect("first call");
        assert_eq!(first.id, "call_a");
        assert_eq!(first.function.name, "get_weather");
        assert_eq!(
            first.function.arguments,
            serde_json::json!({"location": "Paris"})
        );
        let second = calls.get(1).expect("second call");
        assert_eq!(second.id, "call_b");
        assert_eq!(second.function.name, "get_time");
        assert_eq!(
            second.function.arguments,
            serde_json::json!({"zone": "UTC"})
        );
    }

    #[tokio::test]
    async fn tool_call_flushes_at_content_block_stop() {
        // The call must not wait for MessageStop: closing the block emits it.
        let (items, state) = run_events(vec![
            tool_start_event(0, "call_a", "get_weather"),
            tool_delta_event(0, "{}"),
            block_stop_event(0),
        ]);

        assert!(state.tool_calls.is_empty(), "state must be cleared at stop");
        let (calls, errors) = assembled(items).await;
        assert!(errors.is_empty());
        assert_eq!(calls.len(), 1);
    }

    #[tokio::test]
    async fn message_stop_flushes_stragglers_missing_a_block_stop() {
        // Defensive path: a stream that omits ContentBlockStop still delivers
        // every accumulated call at MessageStop, in block order.
        let (items, _state) = run_events(vec![
            tool_start_event(0, "call_a", "get_weather"),
            tool_delta_event(0, "{\"location\":\"Paris\"}"),
            tool_start_event(1, "call_b", "get_time"),
            tool_delta_event(1, "{\"zone\":\"UTC\"}"),
            message_stop_event(aws_bedrock::StopReason::ToolUse),
        ]);

        let (calls, errors) = assembled(items).await;
        assert!(errors.is_empty());
        assert_eq!(calls.len(), 2);
        assert_eq!(calls.first().expect("first call").id, "call_a");
        assert_eq!(calls.get(1).expect("second call").id, "call_b");
    }

    #[tokio::test]
    async fn text_after_closed_tool_block_is_delivered() {
        // A text block following a closed tool-use block used to be discarded
        // because the single tool slot was never cleared.
        let (items, _state) = run_events(vec![
            tool_start_event(0, "call_a", "get_weather"),
            tool_delta_event(0, "{}"),
            block_stop_event(0),
            text_delta_event(1, "Checking the weather now."),
            block_stop_event(1),
            message_stop_event(aws_bedrock::StopReason::EndTurn),
        ]);

        let texts: Vec<&str> = items
            .iter()
            .filter_map(|item| match item {
                Ok(RawStreamingChoice::Message(text)) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(texts, vec!["Checking the weather now."]);
        let (calls, errors) = assembled(items).await;
        assert!(errors.is_empty());
        assert_eq!(calls.len(), 1);
    }

    #[tokio::test]
    async fn malformed_tool_json_surfaces_an_error_item() {
        // Malformed accumulated input must not be silently dropped while the
        // terminal still claims tool use: the consumer gets an error item.
        let (items, _state) = run_events(vec![
            tool_start_event(0, "call_a", "get_weather"),
            tool_delta_event(0, "{\"location\": not-json"),
            block_stop_event(0),
            message_stop_event(aws_bedrock::StopReason::ToolUse),
        ]);

        let (calls, errors) = assembled(items).await;
        assert!(calls.is_empty());
        assert!(
            errors.iter().any(|err| matches!(
                err,
                CompletionError::ResponseError(msg) if msg.contains("get_weather")
            )),
            "malformed tool JSON must yield an error item"
        );
    }

    #[tokio::test]
    async fn max_tokens_stop_drops_in_flight_tool_block_without_deltas() {
        // A tool-use block cut off by MaxTokens before any input arrived must
        // produce neither a fabricated `{}`-args call nor an error item; the
        // truncation is signaled by the Length-mapping stop reason on the
        // terminal record.
        let (items, state) = run_events(vec![
            tool_start_event(0, "call_a", "get_weather"),
            message_stop_event(aws_bedrock::StopReason::MaxTokens),
        ]);

        assert!(
            items.iter().all(|item| item.is_ok()),
            "truncation must not surface as an error item"
        );
        assert_eq!(state.final_stop_reason, Some(StopReason::MaxTokens));
        assert_eq!(
            map_stop_reason(&StopReason::MaxTokens),
            rig_core::completion::FinishReason::Length
        );
        assert!(state.tool_calls.is_empty(), "state must be cleared at stop");
        let (calls, errors) = assembled(items).await;
        assert!(calls.is_empty());
        assert!(errors.is_empty(), "truncation must not surface as an error");
    }

    #[tokio::test]
    async fn max_tokens_stop_drops_in_flight_tool_block_with_partial_json() {
        // Same, but with partial JSON accumulated: the malformed input must
        // not be parsed into a spurious Err at MessageStop.
        let (items, state) = run_events(vec![
            tool_start_event(0, "call_a", "get_weather"),
            tool_delta_event(0, "{\"location\":\"Par"),
            message_stop_event(aws_bedrock::StopReason::MaxTokens),
        ]);

        assert!(
            items.iter().all(|item| item.is_ok()),
            "a truncated partial-JSON block must not yield an error item"
        );
        assert_eq!(state.final_stop_reason, Some(StopReason::MaxTokens));
        assert!(state.tool_calls.is_empty(), "state must be cleared at stop");
        let (calls, errors) = assembled(items).await;
        assert!(calls.is_empty());
        assert!(errors.is_empty(), "no spurious Err from the partial block");
    }

    #[tokio::test]
    async fn empty_tool_input_becomes_empty_object() {
        // A tool with no parameters streams no input deltas at all.
        let (items, _state) = run_events(vec![
            tool_start_event(0, "call_a", "ping"),
            block_stop_event(0),
            message_stop_event(aws_bedrock::StopReason::ToolUse),
        ]);

        let (calls, errors) = assembled(items).await;
        assert!(errors.is_empty());
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls.first().expect("call").function.arguments,
            serde_json::json!({})
        );
    }
}

#[cfg(test)]
mod response_identity_tests {
    use super::*;

    /// Blocking/streaming parity (rig#2265): the streaming terminal's AWS
    /// request id — stamped from the SDK operation output, the same source
    /// the unary surface reads — normalizes into
    /// `StreamFinal.provider_request_id`.
    #[test]
    fn streaming_terminal_request_id_normalizes_into_stream_final() {
        let response = BedrockStreamingResponse {
            usage: None,
            stop_reason: Some(StopReason::EndTurn),
            provider_request_id: Some("aws-req-1".to_string()),
        };

        let usage = (&response).into();
        let terminal = rig_core::streaming::StreamFinal::new(PROVIDER_NAME, usage)
            .with_optional_provider_request_id(response.provider_request_id.clone())
            .with_optional_finish_reason(response.stop_reason.as_ref().map(map_stop_reason));
        assert_eq!(terminal.provider_request_id.as_deref(), Some("aws-req-1"));

        // And a response without one stays None — never an error.
        let without = BedrockStreamingResponse {
            usage: None,
            stop_reason: None,
            provider_request_id: None,
        };
        let terminal = rig_core::streaming::StreamFinal::new(PROVIDER_NAME, (&without).into())
            .with_optional_provider_request_id(without.provider_request_id.clone());
        assert_eq!(terminal.provider_request_id, None);
    }
}
