use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::completion::{
    AnthropicCompatibleProvider, AnthropicCompletionRequest, Content, GenericCompletionModel,
    Usage, anthropic_usage_totals, map_finish_reason,
};
use crate::completion::{CompletionError, CompletionRequest};
use crate::http_client::sse::GenericEventSource;
use crate::http_client::{self, HttpClientExt};
use crate::message::ReasoningContent;
use crate::providers::internal::adapter::{AdapterOutput, WireAdapter, WireFrame};
use crate::providers::internal::sse_transport::{
    OpenLog, SseTransportOptions, open_wire_stream, skip_blank_frames,
};
use crate::providers::internal::wire::{self, WireEvent};
use crate::streaming::{
    self, MintKind, RawStreamingChoice, RawStreamingResult, StreamFinal, StreamPartId,
    ToolCallDeltaContent, ToolInputEnd, UnparseableToolInput,
};
use crate::telemetry::{CompletionOperation, SpanCombinator};
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use std::collections::HashMap;

/// Patch the shared typed request into the Anthropic *streaming* request body.
///
/// The body derives from the *same* typed [`AnthropicCompletionRequest`] the
/// blocking path builds (in `completion.rs`), rather than being re-assembled by
/// hand. The previous hand-rolled `json!` body had drifted from the blocking one
/// and silently dropped `output_schema` (structured-output config); reaching for
/// the typed request fixes that and keeps the two in lockstep. Only the two
/// streaming-only differences documented below are applied here.
fn streaming_body(request: &AnthropicCompletionRequest) -> Result<Value, CompletionError> {
    let mut body = serde_json::to_value(request)?;
    if let Some(map) = body.as_object_mut() {
        // `AnthropicCompletionRequest` has no `stream` field (the blocking path
        // omits it, defaulting to non-streaming); set it for the streaming endpoint.
        map.insert("stream".to_string(), Value::Bool(true));

        // Preserve the streaming path's long-standing `tool_choice` shape, which
        // emitted `tool_choice` *iff* a non-empty tool set was advertised (Anthropic
        // rejects `tool_choice` without `tools`). The blocking typed request instead
        // serializes any caller-set `tool_choice` regardless of tools and omits it
        // when unset, so reconcile here:
        //   - tools present, choice unset -> add the explicit `auto` the streaming
        //     wire has always carried (equivalent to Anthropic's default);
        //   - tools absent -> drop a caller-set `tool_choice` that would otherwise
        //     be sent without `tools` and rejected.
        if map.contains_key("tools") {
            map.entry("tool_choice")
                .or_insert_with(|| json!({ "type": "auto" }));
        } else {
            map.remove("tool_choice");
        }
    }

    Ok(body)
}

/// The `type` values this client models on the Anthropic Messages SSE wire.
///
/// [`classify_tagged_frame`] dispatches on this list: a frame whose `type` is
/// outside it classifies `Unknown` (driver policy: warn + skip), while a
/// listed type must pass the full [`StreamingEvent`] decode or classify
/// `Corrupt`. There is no `#[serde(other)]` fallback — policy lives in the
/// classify layer, never in serde. The one modeled exception is a novel
/// *nested* delta type inside `content_block_delta`, which decodes to
/// [`ContentDelta::Unknown`] (a warned no-op) via its hand-written dispatch.
const KNOWN_EVENT_TYPES: &[&str] = &[
    "message_start",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
    "message_delta",
    "message_stop",
    "ping",
    "error",
];

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamingEvent {
    MessageStart {
        /// Anthropic-compatible relays (Bedrock's Messages passthrough) can
        /// emit `message_start` with a null `message`; `None` is a no-op
        /// rather than a corrupt frame.
        #[serde(default)]
        message: Option<MessageStart>,
    },
    ContentBlockStart {
        index: usize,
        content_block: Content,
    },
    ContentBlockDelta {
        index: usize,
        delta: ContentDelta,
    },
    ContentBlockStop {
        index: usize,
    },
    MessageDelta {
        delta: MessageDelta,
        usage: PartialUsage,
    },
    MessageStop,
    /// Keep-alive; a Known no-op, not an unknown event to warn about.
    Ping,
    /// Anthropic's top-level error envelope (`{"type":"error","error":{...}}`,
    /// e.g. `overloaded_error`). A modeled event, not an unknown to warn-skip:
    /// it surfaces as a provider error like every other family's error
    /// envelope. The payload stays a raw `Value` so every provider field
    /// (type, message, extras) survives into the error body.
    Error {
        error: serde_json::Value,
    },
}

#[derive(Debug, Deserialize)]
pub struct MessageStart {
    pub id: String,
    pub role: String,
    pub content: Vec<Content>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

#[derive(Debug)]
pub enum ContentDelta {
    TextDelta {
        text: String,
    },
    InputJsonDelta {
        partial_json: String,
    },
    ThinkingDelta {
        thinking: String,
    },
    SignatureDelta {
        signature: String,
    },
    CitationsDelta {
        citation: super::completion::Citation,
    },
    /// Any nested delta type this client doesn't model. Anthropic's
    /// versioning policy reserves the right to add new delta types without
    /// notice, so an unmodeled nested tag must not fail the whole
    /// `content_block_delta` frame (which would classify it `Corrupt` and
    /// surface an `Err` item per frame). It decodes to a no-op, warned at the
    /// interpret site — the same shape as
    /// [`ContentPartChunkPart::Unknown`](crate::providers::openai::responses_api::streaming::ContentPartChunkPart).
    Unknown(serde_json::Value),
}

/// Hand-written tag dispatch instead of a trailing `#[serde(untagged)]`
/// variant: on an internally-tagged enum the untagged fallback also swallows
/// a *known* tag with an invalid payload, silently demoting a data-level
/// defect to a skippable unknown delta. Here a known delta tag must decode
/// fully or error (the frame classifies `Corrupt`); only an unmodeled (or
/// absent) tag falls back to [`ContentDelta::Unknown`], preserving the value
/// verbatim. Same pattern as `ContentPartChunkPart`'s hand dispatch in
/// `openai/responses_api/streaming.rs`.
impl<'de> Deserialize<'de> for ContentDelta {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        // A non-object delta is a data-level defect of the tagged shape, not
        // an unmodeled delta kind: it errors (classifying the frame
        // `Corrupt`) instead of degrading to an `Unknown` no-op — the
        // conformance corpus pins `"delta": 42` as Corrupt.
        if !value.is_object() {
            return Err(serde::de::Error::custom("content delta must be an object"));
        }
        let str_field = |tag: &str, field: &str| -> Result<String, D::Error> {
            value
                .get(field)
                .and_then(serde_json::Value::as_str)
                .map(ToOwned::to_owned)
                .ok_or_else(|| {
                    serde::de::Error::custom(format!(
                        "`{tag}` content delta is missing a string `{field}` field"
                    ))
                })
        };
        match value.get("type").cloned() {
            Some(serde_json::Value::String(tag)) => match tag.as_str() {
                "text_delta" => Ok(Self::TextDelta {
                    text: str_field("text_delta", "text")?,
                }),
                "input_json_delta" => Ok(Self::InputJsonDelta {
                    partial_json: str_field("input_json_delta", "partial_json")?,
                }),
                "thinking_delta" => Ok(Self::ThinkingDelta {
                    thinking: str_field("thinking_delta", "thinking")?,
                }),
                "signature_delta" => Ok(Self::SignatureDelta {
                    signature: str_field("signature_delta", "signature")?,
                }),
                "citations_delta" => {
                    let citation = value.get("citation").cloned().ok_or_else(|| {
                        serde::de::Error::custom(
                            "`citations_delta` content delta is missing a `citation` field",
                        )
                    })?;
                    Ok(Self::CitationsDelta {
                        citation: serde_json::from_value(citation)
                            .map_err(serde::de::Error::custom)?,
                    })
                }
                _ => Ok(Self::Unknown(value)),
            },
            Some(_) => Err(serde::de::Error::custom(
                "content delta `type` must be a string",
            )),
            // A content delta without a `type` is malformed, not novel: an
            // untagged text delta from a compat gateway silently skipping
            // here would yield a successful *empty* completion. Corrupt
            // surfaces in-band and the stream keeps consuming.
            None => Err(serde::de::Error::custom(
                "content delta is missing a `type` field",
            )),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct MessageDelta {
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
}

#[derive(Debug, Deserialize, Clone, Serialize, Default)]
pub struct PartialUsage {
    pub output_tokens: usize,
    #[serde(default)]
    pub input_tokens: Option<usize>,
    #[serde(default)]
    pub cache_creation_input_tokens: Option<u64>,
    /// Per-TTL breakdown of `cache_creation_input_tokens`. Anthropic reports
    /// it on `message_start`, not the terminal `message_delta`; the adapter
    /// carries it forward onto the terminal usage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation: Option<super::completion::CacheCreation>,
    #[serde(default)]
    pub cache_read_input_tokens: Option<u64>,
    /// Breakdown of `output_tokens`. Anthropic reports it on the terminal
    /// `message_delta` — the frame that also carries the final `output_tokens`
    /// — not on `message_start`, so unlike `cache_creation` it needs no
    /// carry-forward.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<super::completion::OutputTokensDetails>,
}

impl From<&PartialUsage> for crate::completion::Usage {
    fn from(value: &PartialUsage) -> crate::completion::Usage {
        anthropic_usage_totals(
            value.input_tokens.unwrap_or_default() as u64,
            value.output_tokens as u64,
            value.cache_read_input_tokens,
            value.cache_creation_input_tokens,
            value.output_tokens_details,
        )
    }
}

impl From<PartialUsage> for crate::completion::Usage {
    fn from(value: PartialUsage) -> crate::completion::Usage {
        (&value).into()
    }
}

// Client tool-call fragment assembly lives in the shared accumulator
// (`PartsAccumulator::tool_input_*`); the adapter tracks only the open block's
// wire id. Server tool use keeps local state because its assembled payload
// becomes text-block metadata (`ANTHROPIC_RAW_CONTENT_KEY`), not a tool call.
struct ServerToolUseState {
    name: String,
    id: String,
    initial_input: Value,
    input_json: String,
}

#[derive(Default)]
struct ThinkingState {
    /// Signature assembled from this block's `signature_delta`s. Only the
    /// signature is adapter-side state — the wire fragments it across
    /// deltas and delivers no completed form, so the adapter assembles it
    /// for the block's end event. Thinking TEXT accumulates in the shared
    /// accumulator via `ReasoningDelta`s; no restatement buffer exists.
    signature: String,
    /// The `signature` `content_block_start` opened the block with.
    ///
    /// Recorded traffic always carries the empty string here and delivers the
    /// whole signature by delta, so this is kept as a FALLBACK for a block
    /// that never sends a delta — not as a prefix the deltas extend. A wire
    /// that ever delivered the signature up front still round-trips; a
    /// delta-bearing block never double-counts the opening value.
    initial_signature: String,
}

impl ThinkingState {
    /// The block's completed signature: deltas win over the opening value,
    /// and an absent signature is `None`.
    fn into_signature(self) -> Option<String> {
        let signature = if self.signature.is_empty() {
            self.initial_signature
        } else {
            self.signature
        };
        (!signature.is_empty()).then_some(signature)
    }
}

/// The Anthropic Messages SSE wire as a [`WireAdapter`].
///
/// Holds the per-stream assembly state (open tool call, server tool uses,
/// open thinking block, terminal metadata); frame-triage policy lives in
/// [`run_wire_stream`](crate::providers::internal::adapter::run_wire_stream),
/// not here.
#[derive(Default)]
struct AnthropicAdapter {
    /// Wire id of the open client tool-use block, when one is streaming.
    current_tool_call: Option<String>,
    server_tool_uses: HashMap<usize, ServerToolUseState>,
    current_thinking: Option<ThinkingState>,
    input_tokens: u64,
    /// Per-TTL cache-write breakdown from `message_start`; the terminal
    /// `message_delta` usage omits it.
    cache_creation: Option<super::completion::CacheCreation>,
    message_id: Option<String>,
    response_model: Option<String>,
    /// A provider `error` event ended the turn; later frames are dead — the
    /// provider aborted, and interpreting more output (or a terminal) would
    /// dress the failure up as a completed turn.
    failed: bool,
}

impl WireAdapter for AnthropicAdapter {
    type Frame = WireFrame;
    type Event = StreamingEvent;
    type Response = StreamingCompletionResponse;

    fn classify(&self, frame: WireFrame) -> WireEvent<StreamingEvent> {
        wire::classify_tagged_frame(&frame.as_str(), "type", |event_type| {
            KNOWN_EVENT_TYPES.contains(&event_type)
        })
    }

    fn interpret(&mut self, event: StreamingEvent, out: &mut AdapterOutput<Self::Response>) {
        if self.failed {
            return;
        }

        match &event {
            StreamingEvent::MessageStart { message } => {
                // Bedrock-compat quirk: a `message_start` without a message
                // body is a no-op, not an error.
                let Some(message) = message else { return };
                self.input_tokens = message.usage.input_tokens;
                self.cache_creation
                    .clone_from(&message.usage.cache_creation);
                self.message_id = Some(message.id.clone());
                self.response_model = Some(message.model.clone());

                let span = tracing::Span::current();
                span.record("gen_ai.response.id", &message.id);
                span.record("gen_ai.response.model", &message.model);
                return;
            }
            StreamingEvent::MessageDelta { delta, usage } => {
                // Only a `message_delta` carrying a stop reason is the
                // provider's genuine terminal; without one it is a no-op.
                let Some(reason) = delta.stop_reason.as_ref() else {
                    return;
                };
                // cache_creation_input_tokens and cache_read_input_tokens are
                // cumulative totals on message_delta.usage per the Anthropic
                // streaming API spec — use them directly.
                //
                // `input_tokens` prefers the terminal `message_delta` and falls
                // back to `message_start`.
                //
                // Anthropic proper sends the count on *both* frames and they
                // agree (every recorded cassette under
                // `tests/cassettes/anthropic/` reporting it on the delta reports
                // the same value on the start), so the preference is what runs
                // there and the fallback is inert. The fallback covers the
                // reverse split — a delta that omits the count, leaving the one
                // `message_start` reported.
                //
                // It does *not* rescue the Bedrock-compat body-less
                // `message_start`: that shape returns early above without
                // setting `self.input_tokens`, so the fallback yields
                // `Some(0)`. Preferring the delta is what carries a real count
                // there — do not drop the preference on the theory that the
                // fallback covers that case.
                //
                // Anthropic-*compatible* gateways do not all agree. OpenRouter's
                // Messages endpoint can send `input_tokens: 0` on
                // `message_start` and the real count on `message_delta`
                // (recorded in `gateway_message_delta_metadata`, which OpenRouter
                // served from an Amazon Bedrock upstream — the split follows what
                // it routes to, so it is not every response from that endpoint).
                // Without this preference such a turn surfaces a silent
                // `Usage { input_tokens: 0 }` — worse than a missing value for a
                // consumer sizing its context window from it.
                //
                // Zero on the delta is read as "not reported" so a gateway with
                // the inverse split cannot erase a count `message_start` got
                // right. Note this is a heuristic, not an invariant: a fully
                // cache-hit prompt legitimately bills zero *uncached* input
                // tokens, and its real size lives in the cache fields. Nothing
                // is lost today because both frames then carry the same zero and
                // the fallback yields it anyway — but do not extend the `> 0`
                // filter to the `message_start` side or the cache fields, where
                // a genuine zero would be discarded.
                let usage = PartialUsage {
                    output_tokens: usage.output_tokens,
                    input_tokens: usage
                        .input_tokens
                        .filter(|tokens| *tokens > 0)
                        .or_else(|| usize::try_from(self.input_tokens).ok()),
                    cache_creation_input_tokens: usage.cache_creation_input_tokens,
                    cache_creation: usage.cache_creation.or(self.cache_creation),
                    cache_read_input_tokens: usage.cache_read_input_tokens,
                    // Taken from this frame alone, with no `message_start`
                    // fallback: unlike `cache_creation`, Anthropic reports the
                    // output-token breakdown on the terminal `message_delta`,
                    // the same frame that carries the final `output_tokens` it
                    // breaks down. `message_start` has none to carry forward.
                    output_tokens_details: usage.output_tokens_details,
                };

                let span = tracing::Span::current();
                span.record_token_usage(&crate::completion::Usage::from(&usage));
                out.push(Ok(RawStreamingChoice::FinalResponse(
                    StreamingCompletionResponse {
                        usage,
                        stop_reason: Some(reason.clone()),
                        // Rides the same `message_delta` as the stop reason,
                        // and only that frame carries it: `message_start`
                        // always opens with `null`.
                        stop_sequence: delta.stop_sequence.clone(),
                        message_id: self.message_id.clone(),
                        model: self.response_model.clone(),
                        // Stamped by the transport layer; the adapter never
                        // sees connection headers.
                        provider_request_id: None,
                    },
                )));
                return;
            }
            StreamingEvent::Error { error } => {
                // The provider aborted the turn in-band. Preserve the full
                // error envelope (code + message + extras) as the error body,
                // matching the interactions wire's handling; the stream
                // carries it as an in-band `Err` item, and EOF without
                // `message_delta` then withholds the terminal record.
                self.failed = true;
                let body = serde_json::json!({ "type": "error", "error": error }).to_string();
                out.push(Err(crate::provider_response::completion_error_from_body(
                    body,
                )));
                return;
            }
            _ => {}
        }

        if let Some(result) = handle_event(
            &event,
            &mut self.current_tool_call,
            &mut self.server_tool_uses,
            &mut self.current_thinking,
        ) {
            out.push(result);
        }
    }

    fn finish(&mut self, _out: &mut AdapterOutput<Self::Response>) {
        // EOF without `message_delta` is truncation: open blocks stay
        // partial, and no terminal record may be synthesized.
    }

    fn is_finished(&self) -> bool {
        // A provider `error` event is the wire's own terminal failure:
        // `interpret` already pushed the in-band `Err`, so the driver must
        // stop reading — a later modeled frame (e.g. a stray `message_delta`)
        // would otherwise dress the aborted turn up as a completed one.
        self.failed
    }
}

/// Anthropic's own terminal stream record, as returned by
/// [`GenericCompletionModel::raw_stream`].
///
/// [`crate::completion::CompletionModel::stream`] maps this once into the
/// normalized [`StreamFinal`]; callers who want the provider-native shape read
/// it here instead.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct StreamingCompletionResponse {
    /// Token usage carried by the terminal `message_delta` event.
    pub usage: PartialUsage,
    /// Anthropic's `stop_reason`, verbatim, when the stream reported one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
    /// Which of the caller's `stop_sequences` actually fired, verbatim, when
    /// the terminal `message_delta` reported one.
    ///
    /// `stop_reason: "stop_sequence"` says only *that* a sequence matched;
    /// the sequence itself is the part a caller branches on, and Anthropic
    /// strips it from the text, so the wire is its only source. The blocking
    /// twin has carried it on
    /// [`CompletionResponse::stop_sequence`](super::completion::CompletionResponse::stop_sequence)
    /// all along — the streamed record dropped it after parsing, so the same
    /// request answered strictly less when streamed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
    /// The `message_start` message ID, when the stream reported one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message_id: Option<String>,
    /// The model named by `message_start`, when the stream reported one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// The transport request id from the SSE connection's `request-id`
    /// response header — not part of any stream frame; stamped by the
    /// transport. `None` when the provider did not report one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_request_id: Option<String>,
}

/// Normalize an Anthropic terminal stream record.
///
/// The provider descriptor name is an *input* rather than a constant: the
/// Anthropic Messages stream format is shared by every Anthropic-compatible
/// provider, so baking in `"anthropic"` here would mislabel all of them.
impl From<(&str, StreamingCompletionResponse)> for StreamFinal {
    fn from((provider, response): (&str, StreamingCompletionResponse)) -> Self {
        StreamFinal::new(provider, crate::completion::Usage::from(&response.usage))
            .with_optional_finish_reason(response.stop_reason.as_deref().map(map_finish_reason))
            .with_optional_message_id(response.message_id)
            .with_optional_provider_request_id(response.provider_request_id)
            .with_optional_model(response.model)
    }
}

impl<Ext, T> GenericCompletionModel<Ext, T>
where
    T: HttpClientExt + Clone + 'static,
    Ext: AnthropicCompatibleProvider + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Open a stream whose terminal record stays Anthropic-native.
    ///
    /// This is the escape hatch for provider-specific terminal fields rig does
    /// not normalize. It shares the request builder, transport, telemetry, and
    /// error handling with
    /// [`CompletionModel::stream`](crate::completion::CompletionModel::stream),
    /// which calls it and then maps the terminal record once through
    /// [`crate::streaming::normalize_stream`] — one network request either way.
    pub async fn raw_stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<RawStreamingResult<StreamingCompletionResponse>, CompletionError> {
        let (span, request) =
            self.prepare_request(completion_request, CompletionOperation::ChatStreaming)?;

        // Logged after the streaming-only patches, not on the shared typed
        // request: `stream` and the reconciled `tool_choice` are exactly what
        // makes this body differ from the blocking one.
        let body = streaming_body(&request)?;
        crate::providers::internal::trace_json(
            crate::providers::internal::LogTarget::Completions,
            "Anthropic completion request",
            &body,
        );

        let body: Vec<u8> = serde_json::to_vec(&body)?;

        let req = self
            .client
            .post("/v1/messages")?
            .body(body)
            .map_err(http_client::Error::Protocol)?;

        let event_source = GenericEventSource::new(self.client.clone(), req);
        let (event_source, request_id_slot) = match Ext::REQUEST_ID_HEADER {
            Some(header) => {
                let (event_source, slot) = event_source.capture_request_id(header);
                (event_source, Some(slot))
            }
            None => (event_source, None),
        };

        // Anthropic's loop historically had no separate `StreamEnded` arm and
        // no transport-error log: `StreamEnded` folds into the generic error
        // mapping, preserved via the options below.
        let stream = open_wire_stream(
            event_source,
            SseTransportOptions {
                open_log: OpenLog::Silent,
                stream_ended_is_error: true,
                log_transport_errors: false,
            },
            skip_blank_frames,
            AnthropicAdapter::default(),
            span,
        );
        Ok(
            crate::providers::internal::sse_transport::stamp_terminal_request_id(
                stream,
                request_id_slot,
                Ext::REQUEST_ID_HEADER,
                |response, id| response.provider_request_id = Some(id),
            ),
        )
    }

    pub(crate) async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<streaming::StreamingCompletionResponse, CompletionError> {
        let stream = self.raw_stream(completion_request).await?;
        let normalized = streaming::normalize_stream(stream, |response| {
            Ok(StreamFinal::from((Ext::PROVIDER_NAME, response)))
        });

        Ok(streaming::StreamingCompletionResponse::stream(
            Ext::PROVIDER_NAME,
            normalized,
        ))
    }
}

fn handle_event(
    event: &StreamingEvent,
    current_tool_call: &mut Option<String>,
    server_tool_uses: &mut HashMap<usize, ServerToolUseState>,
    current_thinking: &mut Option<ThinkingState>,
) -> Option<Result<RawStreamingChoice<StreamingCompletionResponse>, CompletionError>> {
    match event {
        StreamingEvent::ContentBlockDelta { index, delta } => match delta {
            ContentDelta::TextDelta { text } => {
                if current_tool_call.is_none() {
                    return Some(Ok(RawStreamingChoice::Message(text.clone())));
                }
                None
            }
            ContentDelta::InputJsonDelta { partial_json } => {
                if let Some(server_tool_use) = server_tool_uses.get_mut(index) {
                    server_tool_use.input_json.push_str(partial_json);
                    return None;
                }

                if let Some(id) = current_tool_call {
                    // Emit the delta so UI can show progress; the shared
                    // accumulator assembles the fragments.
                    return Some(Ok(RawStreamingChoice::ToolCallDelta {
                        id: StreamPartId::wire(id.clone()),
                        content: ToolCallDeltaContent::Delta(partial_json.clone()),
                    }));
                }
                None
            }
            ContentDelta::ThinkingDelta { thinking } => {
                current_thinking.get_or_insert_with(ThinkingState::default);

                Some(Ok(RawStreamingChoice::ReasoningDelta {
                    // Anthropic has no reasoning item id; the content-block
                    // index is stable across a block's deltas and its stop.
                    id: MintKind::Block.for_wire_index(*index as u64),
                    provider_id: None,
                    reasoning: thinking.clone(),
                }))
            }
            ContentDelta::SignatureDelta { signature } => {
                current_thinking
                    .get_or_insert_with(ThinkingState::default)
                    .signature
                    .push_str(signature);

                // Wire quirk: the signature is not emitted as its own chunk —
                // it closes the thinking block, riding on the completed
                // `Reasoning` the `content_block_stop` restatement emits.
                None
            }
            ContentDelta::CitationsDelta { citation } => {
                crate::message::AdditionalParams::from_entries([("citations", json!([citation]))])
                    .map(|params| Ok(RawStreamingChoice::TextAdditionalParams(params)))
            }
            ContentDelta::Unknown(value) => {
                // Structural metadata only: a novel delta type can carry
                // model output, which must not leak into production WARN
                // logs (same policy as the adapter's unknown-event warn).
                tracing::warn!(
                    delta_type = value.get("type").and_then(serde_json::Value::as_str),
                    "skipping unrecognized Anthropic content delta type"
                );
                None
            }
        },
        StreamingEvent::ContentBlockStart {
            index,
            content_block,
        } => match content_block {
            // Keep this destructuring exhaustive so new wire fields force an
            // explicit capture-or-drop decision: block-start `text` arrives
            // via the deltas, and `cache_control` is a request-side
            // directive — both deliberately dropped here.
            Content::Text {
                text: _,
                citations,
                cache_control: _,
            } => {
                let additional_params = crate::message::AdditionalParams::from_entries(
                    (!citations.is_empty()).then(|| ("citations", json!(citations))),
                );
                Some(Ok(RawStreamingChoice::TextStart {
                    // Anthropic has no text item id; the content-block index
                    // is stable for the block's lifetime.
                    id: MintKind::Block.for_wire_index(*index as u64),
                    additional_params,
                }))
            }
            Content::ServerToolUse { id, name, input } => {
                server_tool_uses.insert(
                    *index,
                    ServerToolUseState {
                        name: name.clone(),
                        id: id.clone(),
                        initial_input: input.clone(),
                        input_json: String::new(),
                    },
                );
                None
            }
            raw @ (Content::WebSearchToolResult { .. }
            | Content::CodeExecutionToolResult { .. }) => Some(Ok(RawStreamingChoice::TextStart {
                id: MintKind::Block.for_wire_index(*index as u64),
                additional_params: crate::message::AdditionalParams::from_entries([(
                    super::completion::ANTHROPIC_RAW_CONTENT_KEY,
                    json!(raw),
                )]),
            })),
            Content::ToolUse { id, name, .. } => {
                *current_tool_call = Some(id.clone());
                Some(Ok(RawStreamingChoice::ToolCallDelta {
                    id: StreamPartId::wire(id.clone()),
                    content: ToolCallDeltaContent::Name(name.clone()),
                }))
            }
            Content::Thinking {
                thinking,
                signature,
            } => {
                // `content_block_start` opens the block with its initial
                // payload; the old `..` discarded both fields. Adaptive
                // thinking opens with an empty `thinking`, emits no
                // `thinking_delta` at all, and delivers the whole signature
                // by `signature_delta` — so the block's only content is a
                // signature, which `content_block_stop` must still restate.
                *current_thinking = Some(ThinkingState {
                    signature: String::new(),
                    initial_signature: signature.clone().unwrap_or_default(),
                });
                // The opening payload's text is a delta like any other; the
                // shared accumulator owns the block's text.
                (!thinking.is_empty()).then(|| {
                    Ok(RawStreamingChoice::ReasoningDelta {
                        id: MintKind::Block.for_wire_index(*index as u64),
                        provider_id: None,
                        reasoning: thinking.clone(),
                    })
                })
            }
            Content::RedactedThinking { data } => Some(Ok(RawStreamingChoice::Reasoning {
                // Derive the key from the content-block index (no wire id).
                id: MintKind::Block.for_wire_index(*index as u64),
                provider_id: None,
                content: ReasoningContent::Redacted { data: data.clone() },
            })),
            // Handle other content types - they don't need special handling
            _ => None,
        },
        StreamingEvent::ContentBlockStop { index } => {
            // Drop only a wholly empty block. A signature-only thinking block
            // (empty text, complete signature) is the adaptive-thinking wire
            // shape, and its signature is replay-required provider state that
            // Anthropic accepts back verbatim (the paired non-streaming
            // cassette replays that exact empty-text signed block). The
            // non-streaming path has never gated on text, so gating here was
            // a unary/streaming divergence that silently dropped the
            // signature.
            if let Some(thinking_state) = Option::take(current_thinking) {
                // `content_block_stop` is the wire's own lifecycle end: the
                // shared accumulator holds the block's accumulated text, and
                // the end carries the assembled signature (present for
                // signed and adaptive signature-only blocks alike — replay-
                // required provider state either way). A wholly empty block
                // (no deltas, no signature) closes silently.
                return Some(Ok(RawStreamingChoice::ReasoningEnd {
                    id: MintKind::Block.for_wire_index(*index as u64),
                    reasoning: None,
                    signature: thinking_state.into_signature(),
                    // `content_block_stop` is the wire's own end frame, so
                    // even an unsigned block yields its completed event.
                    wire_sent: true,
                }));
            }

            if let Some(server_tool_use) = server_tool_uses.remove(index) {
                let input = if server_tool_use.input_json.is_empty() {
                    if server_tool_use.initial_input.is_null() {
                        json!({})
                    } else {
                        server_tool_use.initial_input
                    }
                } else {
                    match serde_json::from_str(&server_tool_use.input_json) {
                        Ok(json_value) => json_value,
                        Err(e) => return Some(Err(CompletionError::from(e))),
                    }
                };

                return Some(Ok(RawStreamingChoice::TextStart {
                    id: MintKind::Block.for_wire_index(*index as u64),
                    additional_params: crate::message::AdditionalParams::from_entries([(
                        super::completion::ANTHROPIC_RAW_CONTENT_KEY,
                        json!(Content::ServerToolUse {
                            id: server_tool_use.id,
                            name: server_tool_use.name,
                            input,
                        }),
                    )]),
                }));
            }

            // `content_block_stop` promises a complete block: empty input
            // finalizes to `{}`, malformed input surfaces as an error item
            // (`UnparseableToolInput::Error`) in the accumulator.
            Option::take(current_tool_call).map(|id| {
                Ok(RawStreamingChoice::ToolInputEnd(ToolInputEnd::new(
                    id,
                    UnparseableToolInput::Error,
                )))
            })
        }
        // Interpreted by the adapter (`message_start`/`message_delta`/the
        // `error` envelope) or Known no-ops (`message_stop`, `ping`).
        StreamingEvent::MessageStart { .. }
        | StreamingEvent::MessageDelta { .. }
        | StreamingEvent::MessageStop
        | StreamingEvent::Ping
        | StreamingEvent::Error { .. } => None,
    }
}

#[cfg(test)]
mod tests;
