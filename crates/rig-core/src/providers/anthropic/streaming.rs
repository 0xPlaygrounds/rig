use async_stream::stream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tracing::{Level, enabled};
use tracing_futures::Instrument;

use super::completion::{
    AnthropicCompatibleProvider, AnthropicCompletionRequest, AnthropicRequestParams, CacheTtl,
    Content, GenericCompletionModel, Usage, map_finish_reason,
};
use crate::completion::{CompletionError, CompletionRequest};
use crate::http_client::sse::{Event, GenericEventSource};
use crate::http_client::{self, HttpClientExt};
use crate::message::ReasoningContent;
use crate::providers::internal::adapter::{AdapterOutput, WireAdapter, WireFrame, run_wire_stream};
use crate::providers::internal::wire::{self, WireEvent};
use crate::streaming::{
    self, MintKind, PartId, RawStreamingChoice, RawStreamingResult, StreamFinal,
    ToolCallDeltaContent, ToolInputEnd, UnparseableToolInput,
};
use crate::telemetry::{CompletionOperation, CompletionSpanBuilder, SpanCombinator};
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use std::collections::HashMap;

/// Build the Anthropic streaming request body.
///
/// Derives it from the *same* typed [`AnthropicCompletionRequest`] the blocking
/// path builds (in `completion.rs`), rather than re-assembling the body by hand.
/// The previous hand-rolled `json!` body had drifted from the blocking one and
/// silently dropped `output_schema` (structured-output config); reaching for the
/// typed request fixes that and keeps the two in lockstep. The one intentional
/// streaming-only difference — an explicit `tool_choice: auto` when tools are
/// advertised but the caller left the choice unset — is re-applied below so the
/// streaming request bytes stay stable.
fn create_streaming_request_body(
    request_model: String,
    mut completion_request: CompletionRequest,
    max_tokens: u64,
    prompt_caching: bool,
    automatic_caching: bool,
    automatic_caching_ttl: Option<CacheTtl>,
) -> Result<Value, CompletionError> {
    // The typed request's `TryFrom` requires `max_tokens`; feed it the value the
    // caller already resolved (the request's own value, else the model default).
    completion_request.max_tokens = Some(max_tokens);

    let request = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model: &request_model,
        request: completion_request,
        prompt_caching,
        automatic_caching,
        automatic_caching_ttl,
    })?;

    let mut body = serde_json::to_value(&request)?;
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
    #[serde(default)]
    pub cache_read_input_tokens: Option<u64>,
}

impl From<&PartialUsage> for crate::completion::Usage {
    fn from(value: &PartialUsage) -> crate::completion::Usage {
        let mut usage = crate::completion::Usage::new();

        usage.input_tokens = value.input_tokens.unwrap_or_default() as u64;
        usage.output_tokens = value.output_tokens as u64;
        usage.cached_input_tokens = value.cache_read_input_tokens.unwrap_or(0);
        usage.cache_creation_input_tokens = value.cache_creation_input_tokens.unwrap_or(0);
        usage.total_tokens = usage.input_tokens
            + usage.cached_input_tokens
            + usage.cache_creation_input_tokens
            + usage.output_tokens;
        usage
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
    thinking: String,
    /// Signature assembled from this block's `signature_delta`s.
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
    /// The block's completed content as `(text, signature)`: deltas win over
    /// the opening value, and an absent signature is `None`.
    fn into_parts(self) -> (String, Option<String>) {
        let signature = if self.signature.is_empty() {
            self.initial_signature
        } else {
            self.signature
        };

        (self.thinking, (!signature.is_empty()).then_some(signature))
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
                let usage = PartialUsage {
                    output_tokens: usage.output_tokens,
                    input_tokens: usize::try_from(self.input_tokens).ok(),
                    cache_creation_input_tokens: usage.cache_creation_input_tokens,
                    cache_read_input_tokens: usage.cache_read_input_tokens,
                };

                let span = tracing::Span::current();
                span.record_token_usage(&crate::completion::Usage::from(&usage));
                out.push(Ok(RawStreamingChoice::FinalResponse(
                    StreamingCompletionResponse {
                        usage,
                        stop_reason: Some(reason.clone()),
                        message_id: self.message_id.clone(),
                        model: self.response_model.clone(),
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
    /// The `message_start` message ID, when the stream reported one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message_id: Option<String>,
    /// The model named by `message_start`, when the stream reported one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
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
            .with_optional_model(response.model)
    }
}

impl<Ext, T> GenericCompletionModel<Ext, T>
where
    T: HttpClientExt + Clone + Default + 'static,
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
        let request_model = completion_request
            .model
            .clone()
            .unwrap_or_else(|| self.model.clone());
        let span = CompletionSpanBuilder::new(
            Ext::PROVIDER_NAME,
            &request_model,
            CompletionOperation::ChatStreaming,
        )
        .system_instructions(
            completion_request.preamble.as_deref(),
            completion_request.record_telemetry_content,
        )
        .build();
        let max_tokens = if let Some(tokens) = completion_request.max_tokens {
            tokens
        } else if let Some(tokens) = self.default_max_tokens {
            tokens
        } else {
            return Err(CompletionError::RequestError(
                "`max_tokens` must be set for Anthropic".into(),
            ));
        };

        let body = create_streaming_request_body(
            request_model,
            completion_request,
            max_tokens,
            self.prompt_caching,
            self.automatic_caching,
            self.automatic_caching_ttl.clone(),
        )?;

        if enabled!(Level::TRACE) {
            tracing::trace!(
                target: "rig::completions",
                "Anthropic completion request: {}",
                serde_json::to_string_pretty(&body)?
            );
        }

        let body: Vec<u8> = serde_json::to_vec(&body)?;

        let req = self
            .client
            .post("/v1/messages")?
            .body(body)
            .map_err(http_client::Error::Protocol)?;

        let stream = GenericEventSource::new(self.client.clone(), req);

        // Transport layer: SSE events → `WireFrame`s. Byte splitting and
        // framing only — classification and policy live downstream.
        let transport = stream! {
            let mut sse_stream = Box::pin(stream);
            while let Some(sse_result) = sse_stream.next().await {
                match sse_result {
                    Ok(Event::Open) => {}
                    Ok(Event::Message(sse)) => {
                        // Data-less frames (keep-alive comments) carry no
                        // payload and are not wire frames.
                        if sse.data.trim().is_empty() {
                            continue;
                        }
                        yield Ok(WireFrame::Text(sse.data));
                    }
                    Err(e) => {
                        yield Err(CompletionError::from_stream_transport(e));
                        break;
                    }
                }
            }
            // Ensure event source is closed when stream ends
            sse_stream.close();
        };

        let stream: RawStreamingResult<StreamingCompletionResponse> =
            Box::pin(run_wire_stream(transport, AnthropicAdapter::default()).instrument(span));

        Ok(stream)
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
                        id: PartId::wire(id.clone()),
                        content: ToolCallDeltaContent::Delta(partial_json.clone()),
                    }));
                }
                None
            }
            ContentDelta::ThinkingDelta { thinking } => {
                current_thinking
                    .get_or_insert_with(ThinkingState::default)
                    .thinking
                    .push_str(thinking);

                Some(Ok(RawStreamingChoice::ReasoningDelta {
                    // Anthropic has no reasoning item id; the content-block
                    // index is stable across a block's deltas and its stop.
                    id: MintKind::Block.for_wire_index(*index as u64),
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
                Some(Ok(RawStreamingChoice::TextAdditionalParams(json!({
                    "citations": [citation]
                }))))
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
            Content::Text { citations, .. } => {
                let additional_params = (!citations.is_empty()).then(|| {
                    json!({
                        "citations": citations
                    })
                });
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
                additional_params: Some(json!({
                    super::completion::ANTHROPIC_RAW_CONTENT_KEY: raw
                })),
            })),
            Content::ToolUse { id, name, .. } => {
                *current_tool_call = Some(id.clone());
                Some(Ok(RawStreamingChoice::ToolCallDelta {
                    id: PartId::wire(id.clone()),
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
                    thinking: thinking.clone(),
                    signature: String::new(),
                    initial_signature: signature.clone().unwrap_or_default(),
                });
                None
            }
            Content::RedactedThinking { data } => Some(Ok(RawStreamingChoice::Reasoning {
                // Derive identity from the content-block index (no wire id).
                id: MintKind::Block.for_wire_index(*index as u64),
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
                let (text, signature) = thinking_state.into_parts();

                if !(text.is_empty() && signature.is_none()) {
                    return Some(Ok(RawStreamingChoice::Reasoning {
                        // Same block index as this block's ThinkingDeltas, so
                        // the full block supersedes the accumulated deltas.
                        id: MintKind::Block.for_wire_index(*index as u64),
                        content: ReasoningContent::Text { text, signature },
                    }));
                }
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
                    additional_params: Some(json!({
                        super::completion::ANTHROPIC_RAW_CONTENT_KEY: Content::ServerToolUse {
                            id: server_tool_use.id,
                            name: server_tool_use.name,
                            input,
                        }
                    })),
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
mod tests {
    use super::super::completion::{
        CLAUDE_OPUS_4_8, CacheControl, CacheTtl, Message, SystemContent,
        apply_prompt_cache_control, build_tool_definitions, resolve_top_level_cache_control,
    };
    use super::*;
    use crate::OneOrMany;
    use crate::completion::Message as RigMessage;
    use crate::completion::request::Document as RigDocument;
    use crate::streaming::RawStreamingToolCall;
    use async_stream::stream;
    use futures::StreamExt;

    /// Normalize a hand-built Anthropic raw stream exactly as
    /// [`GenericCompletionModel::stream`] does, so aggregation assertions run
    /// against the same terminal-record mapping as the real path.
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    fn to_stream_result(
        stream: impl futures::Stream<
            Item = Result<RawStreamingChoice<StreamingCompletionResponse>, CompletionError>,
        > + Send
        + 'static,
    ) -> crate::streaming::StreamingResult {
        crate::streaming::normalize_stream(Box::pin(stream), |response| {
            Ok(StreamFinal::from(("anthropic", response)))
        })
    }

    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    fn to_stream_result(
        stream: impl futures::Stream<
            Item = Result<RawStreamingChoice<StreamingCompletionResponse>, CompletionError>,
        > + 'static,
    ) -> crate::streaming::StreamingResult {
        crate::streaming::normalize_stream(Box::pin(stream), |response| {
            Ok(StreamFinal::from(("anthropic", response)))
        })
    }

    #[test]
    fn test_streaming_tool_build_marks_final_combined_tool() {
        let mut additional_params = json!({
            "tools": [{
                "name": "provider_tool",
                "description": "Provider tool",
                "input_schema": {"type": "object"}
            }]
        });

        let mut tools = build_tool_definitions(
            vec![crate::completion::ToolDefinition {
                name: "rig_tool".to_string(),
                description: "Rig tool".to_string(),
                parameters: json!({"type": "object", "properties": {}}),
            }],
            &mut additional_params,
        )
        .unwrap();
        let mut system: Vec<SystemContent> = Vec::new();
        let mut messages: Vec<Message> = Vec::new();
        apply_prompt_cache_control(&mut system, &mut messages, &mut tools, true, None).unwrap();

        assert_eq!(tools.len(), 2);
        assert!(tools[0].get("cache_control").is_none());
        assert_eq!(tools[1]["name"], "provider_tool");
        assert_eq!(tools[1]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn streaming_request_keeps_documents_after_leading_system_messages() {
        let request = CompletionRequest {
            model: None,
            preamble: None,
            chat_history: OneOrMany::many(vec![
                RigMessage::system("System prompt"),
                RigMessage::assistant("Earlier assistant turn"),
                RigMessage::system("Mid-conversation instruction"),
                RigMessage::user("Prompt"),
            ])
            .unwrap(),
            documents: vec![RigDocument {
                id: "doc1".to_string(),
                text: "Document text.".to_string(),
                additional_props: Default::default(),
            }],
            tools: vec![],
            temperature: None,
            max_tokens: Some(64),
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        };

        let body = create_streaming_request_body(
            CLAUDE_OPUS_4_8.to_string(),
            request,
            64,
            false,
            false,
            None,
        )
        .expect("streaming request body should build");

        assert_eq!(body["system"][0]["text"], "System prompt");
        assert_eq!(body["system"][1]["text"], "Mid-conversation instruction");
        let messages = body["messages"]
            .as_array()
            .expect("messages should be array");
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0]["role"], "user");
        assert!(
            messages[0].to_string().contains("<file id: doc1>"),
            "document message should follow top-level system: {messages:?}"
        );
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[2]["role"], "user");
        assert_eq!(
            messages
                .iter()
                .filter(|message| message.to_string().contains("<file id: doc1>"))
                .count(),
            1,
            "document message should appear exactly once: {messages:?}"
        );
    }

    #[test]
    fn streaming_body_is_blocking_body_plus_stream_flag_and_carries_output_schema() {
        let schema: schemars::Schema = serde_json::from_value(json!({
            "title": "WeatherResponse",
            "type": "object",
            "properties": { "city": { "type": "string" } }
        }))
        .expect("schema should deserialize");

        let request = CompletionRequest {
            model: None,
            preamble: Some("You are helpful".to_string()),
            chat_history: OneOrMany::one(RigMessage::user("What's the weather?")),
            documents: vec![],
            tools: vec![],
            temperature: Some(0.5),
            max_tokens: Some(64),
            tool_choice: None,
            additional_params: None,
            output_schema: Some(schema),
            record_telemetry_content: false,
        };

        let streaming_body = create_streaming_request_body(
            CLAUDE_OPUS_4_8.to_string(),
            request.clone(),
            64,
            false,
            false,
            None,
        )
        .expect("streaming request body should build");

        // The streaming endpoint flag is set.
        assert_eq!(streaming_body["stream"], serde_json::Value::Bool(true));

        // Regression: `output_schema` now reaches the streaming wire as
        // `output_config` (the hand-rolled body dropped it entirely, so this
        // assertion would have failed before the typed-request unification).
        assert_eq!(
            streaming_body["output_config"]["format"]["type"],
            "json_schema"
        );
        assert!(
            streaming_body["output_config"]["format"]["schema"].is_object(),
            "streaming body must carry the structured-output schema: {streaming_body}"
        );

        // Unification invariant: the streaming body is exactly the blocking body
        // (built via the same typed request) plus `stream: true`. Pins the two
        // wire formats together so a future edit can't reintroduce drift.
        let blocking = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
            model: CLAUDE_OPUS_4_8,
            request,
            prompt_caching: false,
            automatic_caching: false,
            automatic_caching_ttl: None,
        })
        .expect("blocking request body should build");
        let mut expected = serde_json::to_value(&blocking).expect("serialize blocking body");
        expected
            .as_object_mut()
            .expect("body is an object")
            .insert("stream".to_string(), serde_json::Value::Bool(true));

        assert_eq!(streaming_body, expected);
    }

    #[test]
    fn streaming_body_keeps_explicit_tool_choice_auto_when_tools_present_but_unset() {
        let request = CompletionRequest {
            model: None,
            preamble: None,
            chat_history: OneOrMany::one(RigMessage::user("Add 2 and 3")),
            documents: vec![],
            tools: vec![crate::completion::ToolDefinition {
                name: "add".to_string(),
                description: "Add x and y".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": { "x": { "type": "integer" } }
                }),
            }],
            temperature: None,
            max_tokens: Some(64),
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        };

        let body = create_streaming_request_body(
            CLAUDE_OPUS_4_8.to_string(),
            request,
            64,
            false,
            false,
            None,
        )
        .expect("streaming request body should build");

        // Tools advertised + `tool_choice` unset must still carry the explicit
        // `auto` the streaming wire format has always sent (parity with recorded
        // fixtures), even though the blocking typed request omits it.
        assert_eq!(body["tool_choice"], json!({ "type": "auto" }));
        assert!(body["tools"].is_array());
    }

    #[test]
    fn streaming_body_drops_tool_choice_when_no_tools_are_advertised() {
        // The typed request serializes a caller-set `tool_choice` regardless of
        // whether tools are present, but the streaming path has always emitted
        // `tool_choice` *only* alongside a non-empty tool set (Anthropic rejects it
        // otherwise). A `tool_choice` set with no tools must not reach the wire.
        let request = CompletionRequest {
            model: None,
            preamble: None,
            chat_history: OneOrMany::one(RigMessage::user("Hi")),
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: Some(64),
            tool_choice: Some(crate::message::ToolChoice::Auto),
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        };

        let body = create_streaming_request_body(
            CLAUDE_OPUS_4_8.to_string(),
            request,
            64,
            false,
            false,
            None,
        )
        .expect("streaming request body should build");

        assert!(
            body.get("tool_choice").is_none(),
            "tool_choice must be omitted when no tools are advertised: {body}"
        );
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn test_streaming_prompt_cache_control_uses_raw_top_level_ttl() {
        let mut additional_params = json!({
            "cache_control": {"type": "ephemeral", "ttl": "1h"}
        });
        let top_level_cache_control =
            resolve_top_level_cache_control(false, None, &mut additional_params).unwrap();
        let mut tools = build_tool_definitions(
            vec![crate::completion::ToolDefinition {
                name: "rig_tool".to_string(),
                description: "Rig tool".to_string(),
                parameters: json!({"type": "object", "properties": {}}),
            }],
            &mut additional_params,
        )
        .unwrap();
        let mut system = vec![SystemContent::Text {
            text: "System prompt".to_string(),
            cache_control: None,
        }];
        let mut messages: Vec<Message> = Vec::new();

        apply_prompt_cache_control(
            &mut system,
            &mut messages,
            &mut tools,
            true,
            top_level_cache_control.as_ref(),
        )
        .unwrap();

        assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
        assert_eq!(tools[0]["cache_control"]["ttl"], "1h");
        match &system[0] {
            SystemContent::Text {
                cache_control: Some(CacheControl::Ephemeral { ttl }),
                ..
            } => assert_eq!(ttl.as_ref(), Some(&CacheTtl::OneHour)),
            other => panic!("expected system cache_control, got {other:?}"),
        }
        assert!(additional_params.get("cache_control").is_none());
    }

    fn handle_event(
        event: &StreamingEvent,
        current_tool_call: &mut Option<String>,
        current_thinking: &mut Option<ThinkingState>,
    ) -> Option<Result<RawStreamingChoice<StreamingCompletionResponse>, CompletionError>> {
        let mut server_tool_uses = HashMap::new();
        super::handle_event(
            event,
            current_tool_call,
            &mut server_tool_uses,
            current_thinking,
        )
    }

    #[test]
    fn test_thinking_delta_deserialization() {
        let json = r#"{"type": "thinking_delta", "thinking": "Let me think about this..."}"#;
        let delta: ContentDelta = serde_json::from_str(json).unwrap();

        match delta {
            ContentDelta::ThinkingDelta { thinking } => {
                assert_eq!(thinking, "Let me think about this...");
            }
            _ => panic!("Expected ThinkingDelta variant"),
        }
    }

    #[test]
    fn test_signature_delta_deserialization() {
        let json = r#"{"type": "signature_delta", "signature": "abc123def456"}"#;
        let delta: ContentDelta = serde_json::from_str(json).unwrap();

        match delta {
            ContentDelta::SignatureDelta { signature } => {
                assert_eq!(signature, "abc123def456");
            }
            _ => panic!("Expected SignatureDelta variant"),
        }
    }

    #[test]
    fn test_thinking_delta_streaming_event_deserialization() {
        let json = r#"{
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "thinking_delta",
                "thinking": "First, I need to understand the problem."
            }
        }"#;

        let event: StreamingEvent = serde_json::from_str(json).unwrap();

        match event {
            StreamingEvent::ContentBlockDelta { index, delta } => {
                assert_eq!(index, 0);
                match delta {
                    ContentDelta::ThinkingDelta { thinking } => {
                        assert_eq!(thinking, "First, I need to understand the problem.");
                    }
                    _ => panic!("Expected ThinkingDelta"),
                }
            }
            _ => panic!("Expected ContentBlockDelta event"),
        }
    }

    #[test]
    fn test_signature_delta_streaming_event_deserialization() {
        let json = r#"{
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "signature_delta",
                "signature": "ErUBCkYICBgCIkCaGbqC85F4"
            }
        }"#;

        let event: StreamingEvent = serde_json::from_str(json).unwrap();

        match event {
            StreamingEvent::ContentBlockDelta { index, delta } => {
                assert_eq!(index, 0);
                match delta {
                    ContentDelta::SignatureDelta { signature } => {
                        assert_eq!(signature, "ErUBCkYICBgCIkCaGbqC85F4");
                    }
                    _ => panic!("Expected SignatureDelta"),
                }
            }
            _ => panic!("Expected ContentBlockDelta event"),
        }
    }

    #[test]
    fn test_handle_thinking_delta_event() {
        let event = StreamingEvent::ContentBlockDelta {
            index: 0,
            delta: ContentDelta::ThinkingDelta {
                thinking: "Analyzing the request...".to_string(),
            },
        };

        let mut tool_call_state = None;
        let mut thinking_state = None;
        let result = handle_event(&event, &mut tool_call_state, &mut thinking_state);

        assert!(result.is_some());
        let choice = result.unwrap().unwrap();

        match choice {
            RawStreamingChoice::ReasoningDelta { id, reasoning, .. } => {
                assert_eq!(id, crate::streaming::MintKind::Block.for_wire_index(0));
                assert_eq!(reasoning, "Analyzing the request...");
            }
            _ => panic!("Expected ReasoningDelta choice"),
        }

        // Verify thinking state was updated
        assert!(thinking_state.is_some());
        assert_eq!(thinking_state.unwrap().thinking, "Analyzing the request...");
    }

    #[test]
    fn test_handle_signature_delta_event() {
        let event = StreamingEvent::ContentBlockDelta {
            index: 0,
            delta: ContentDelta::SignatureDelta {
                signature: "test_signature".to_string(),
            },
        };

        let mut tool_call_state = None;
        let mut thinking_state = None;
        let result = handle_event(&event, &mut tool_call_state, &mut thinking_state);

        // SignatureDelta should not yield anything (returns None)
        assert!(result.is_none());

        // But signature should be captured in thinking state
        assert!(thinking_state.is_some());
        assert_eq!(thinking_state.unwrap().signature, "test_signature");
    }

    #[test]
    fn test_handle_redacted_thinking_content_block_start_event() {
        let event = StreamingEvent::ContentBlockStart {
            index: 0,
            content_block: Content::RedactedThinking {
                data: "redacted_blob".to_string(),
            },
        };
        let mut tool_call_state = None;
        let mut thinking_state = None;
        let result = handle_event(&event, &mut tool_call_state, &mut thinking_state);

        assert!(result.is_some());
        match result.unwrap().unwrap() {
            RawStreamingChoice::Reasoning {
                content: ReasoningContent::Redacted { data },
                ..
            } => {
                assert_eq!(data, "redacted_blob");
            }
            _ => panic!("Expected Redacted reasoning chunk"),
        }
    }

    /// The adaptive-thinking wire shape, exactly as recorded in
    /// `tests/cassettes/anthropic/opus_4_7/messages_adaptive_thinking_streaming_smoke.yaml`:
    /// `content_block_start` opens the block with an EMPTY `thinking` and an
    /// EMPTY `signature`, a `signature_delta` carries the whole signature, and
    /// no `thinking_delta` ever arrives. The block's only content is its
    /// signature, and it must survive `content_block_stop`.
    #[test]
    fn signature_only_thinking_block_survives_content_block_stop() {
        let mut tool_call_state = None;
        let mut thinking_state = None;

        let start = StreamingEvent::ContentBlockStart {
            index: 0,
            content_block: Content::Thinking {
                thinking: String::new(),
                signature: Some(String::new()),
            },
        };
        assert!(handle_event(&start, &mut tool_call_state, &mut thinking_state).is_none());

        let signature = StreamingEvent::ContentBlockDelta {
            index: 0,
            delta: ContentDelta::SignatureDelta {
                signature: "the_whole_signature".to_string(),
            },
        };
        assert!(handle_event(&signature, &mut tool_call_state, &mut thinking_state).is_none());

        let stop = StreamingEvent::ContentBlockStop { index: 0 };
        let result = handle_event(&stop, &mut tool_call_state, &mut thinking_state)
            .expect("signature-only thinking block must not be dropped")
            .expect("thinking block should not be an error");

        match result {
            RawStreamingChoice::Reasoning {
                id,
                content: ReasoningContent::Text { text, signature },
            } => {
                assert_eq!(id, crate::streaming::MintKind::Block.for_wire_index(0));
                assert_eq!(text, "");
                assert_eq!(signature.as_deref(), Some("the_whole_signature"));
            }
            other => panic!("Expected signed Reasoning chunk, got {other:?}"),
        }
    }

    /// Forward compat: a block that delivers its whole signature on
    /// `content_block_start` and sends no `signature_delta` keeps it.
    #[test]
    fn signature_delivered_only_on_content_block_start_is_kept() {
        let mut tool_call_state = None;
        let mut thinking_state = None;

        let start = StreamingEvent::ContentBlockStart {
            index: 0,
            content_block: Content::Thinking {
                thinking: String::new(),
                signature: Some("up_front_signature".to_string()),
            },
        };
        assert!(handle_event(&start, &mut tool_call_state, &mut thinking_state).is_none());

        let stop = StreamingEvent::ContentBlockStop { index: 0 };
        match handle_event(&stop, &mut tool_call_state, &mut thinking_state)
            .expect("an up-front signature must not be dropped")
            .expect("thinking block should not be an error")
        {
            RawStreamingChoice::Reasoning {
                content: ReasoningContent::Text { text, signature },
                ..
            } => {
                assert_eq!(text, "");
                assert_eq!(signature.as_deref(), Some("up_front_signature"));
            }
            other => panic!("Expected signed Reasoning chunk, got {other:?}"),
        }
    }

    /// The opening `signature` is a fallback, never a prefix the deltas
    /// extend: a delta-bearing block must publish exactly what the deltas
    /// assembled, or the value replayed to Anthropic is corrupt.
    #[test]
    fn signature_deltas_supersede_the_opening_signature() {
        let mut tool_call_state = None;
        let mut thinking_state = None;

        let start = StreamingEvent::ContentBlockStart {
            index: 0,
            content_block: Content::Thinking {
                thinking: String::new(),
                signature: Some("opening".to_string()),
            },
        };
        assert!(handle_event(&start, &mut tool_call_state, &mut thinking_state).is_none());

        for fragment in ["delta_", "assembled"] {
            let signature = StreamingEvent::ContentBlockDelta {
                index: 0,
                delta: ContentDelta::SignatureDelta {
                    signature: fragment.to_string(),
                },
            };
            assert!(handle_event(&signature, &mut tool_call_state, &mut thinking_state).is_none());
        }

        let stop = StreamingEvent::ContentBlockStop { index: 0 };
        match handle_event(&stop, &mut tool_call_state, &mut thinking_state)
            .expect("thinking block should be restated")
            .expect("thinking block should not be an error")
        {
            RawStreamingChoice::Reasoning {
                content: ReasoningContent::Text { signature, .. },
                ..
            } => assert_eq!(signature.as_deref(), Some("delta_assembled")),
            other => panic!("Expected signed Reasoning chunk, got {other:?}"),
        }
    }

    /// `content_block_start` can carry the block's opening text; discarding it
    /// would truncate the restatement the accumulator supersedes deltas with.
    #[test]
    fn thinking_block_start_text_seeds_the_restatement() {
        let mut tool_call_state = None;
        let mut thinking_state = None;

        let start = StreamingEvent::ContentBlockStart {
            index: 2,
            content_block: Content::Thinking {
                thinking: "opening ".to_string(),
                signature: None,
            },
        };
        assert!(handle_event(&start, &mut tool_call_state, &mut thinking_state).is_none());

        let delta = StreamingEvent::ContentBlockDelta {
            index: 2,
            delta: ContentDelta::ThinkingDelta {
                thinking: "rest".to_string(),
            },
        };
        assert!(handle_event(&delta, &mut tool_call_state, &mut thinking_state).is_some());

        let stop = StreamingEvent::ContentBlockStop { index: 2 };
        match handle_event(&stop, &mut tool_call_state, &mut thinking_state)
            .expect("thinking block should be restated")
            .expect("thinking block should not be an error")
        {
            RawStreamingChoice::Reasoning {
                id,
                content: ReasoningContent::Text { text, signature },
            } => {
                assert_eq!(id, crate::streaming::MintKind::Block.for_wire_index(2));
                assert_eq!(text, "opening rest");
                assert_eq!(signature, None);
            }
            other => panic!("Expected Reasoning chunk, got {other:?}"),
        }
    }

    /// A block with neither text nor signature carries nothing to replay.
    #[test]
    fn wholly_empty_thinking_block_is_dropped() {
        let mut tool_call_state = None;
        let mut thinking_state = None;

        let start = StreamingEvent::ContentBlockStart {
            index: 0,
            content_block: Content::Thinking {
                thinking: String::new(),
                signature: None,
            },
        };
        assert!(handle_event(&start, &mut tool_call_state, &mut thinking_state).is_none());

        let stop = StreamingEvent::ContentBlockStop { index: 0 };
        assert!(handle_event(&stop, &mut tool_call_state, &mut thinking_state).is_none());
    }

    #[test]
    fn test_handle_text_delta_event() {
        let event = StreamingEvent::ContentBlockDelta {
            index: 0,
            delta: ContentDelta::TextDelta {
                text: "Hello, world!".to_string(),
            },
        };

        let mut tool_call_state = None;
        let mut thinking_state = None;
        let result = handle_event(&event, &mut tool_call_state, &mut thinking_state);

        assert!(result.is_some());
        let choice = result.unwrap().unwrap();

        match choice {
            RawStreamingChoice::Message(text) => {
                assert_eq!(text, "Hello, world!");
            }
            _ => panic!("Expected Message choice"),
        }
    }

    #[test]
    fn test_handle_text_block_start_event() {
        let event = StreamingEvent::ContentBlockStart {
            index: 0,
            content_block: Content::Text {
                text: String::new(),
                citations: Vec::new(),
                cache_control: None,
            },
        };

        let mut tool_call_state = None;
        let mut thinking_state = None;
        let result = handle_event(&event, &mut tool_call_state, &mut thinking_state);

        assert!(result.is_some());
        let choice = result.unwrap().unwrap();
        assert!(matches!(
            choice,
            RawStreamingChoice::TextStart {
                additional_params: None,
                ..
            }
        ));
    }

    #[test]
    fn test_thinking_delta_does_not_interfere_with_tool_calls() {
        // Thinking deltas should still be processed even if a tool call is in progress
        let event = StreamingEvent::ContentBlockDelta {
            index: 0,
            delta: ContentDelta::ThinkingDelta {
                thinking: "Thinking while tool is active...".to_string(),
            },
        };

        let mut tool_call_state = Some("tool_123".to_string());
        let mut thinking_state = None;

        let result = handle_event(&event, &mut tool_call_state, &mut thinking_state);

        assert!(result.is_some());
        let choice = result.unwrap().unwrap();

        match choice {
            RawStreamingChoice::ReasoningDelta { reasoning, .. } => {
                assert_eq!(reasoning, "Thinking while tool is active...");
            }
            _ => panic!("Expected ReasoningDelta choice"),
        }

        // Tool call state should remain unchanged
        assert!(tool_call_state.is_some());
    }

    #[test]
    fn test_handle_input_json_delta_event() {
        let event = StreamingEvent::ContentBlockDelta {
            index: 0,
            delta: ContentDelta::InputJsonDelta {
                partial_json: "{\"arg\":\"value".to_string(),
            },
        };

        let mut tool_call_state = Some("tool_123".to_string());
        let mut thinking_state = None;

        let result = handle_event(&event, &mut tool_call_state, &mut thinking_state);

        // Should emit a ToolCallDelta
        assert!(result.is_some());
        let choice = result.unwrap().unwrap();

        match choice {
            RawStreamingChoice::ToolCallDelta { id, content } => {
                assert_eq!(id.as_wire(), Some("tool_123"));
                match content {
                    ToolCallDeltaContent::Delta(delta) => assert_eq!(delta, "{\"arg\":\"value"),
                    _ => panic!("Expected Delta content"),
                }
            }
            _ => panic!("Expected ToolCallDelta choice, got {:?}", choice),
        }

        // The open block stays open; assembly of the fragment happens in the
        // shared accumulator.
        assert!(tool_call_state.is_some());
    }

    #[test]
    fn test_tool_call_accumulation_with_multiple_deltas() {
        let mut tool_call_state = Some("tool_123".to_string());
        let mut thinking_state = None;

        // First delta
        let event1 = StreamingEvent::ContentBlockDelta {
            index: 0,
            delta: ContentDelta::InputJsonDelta {
                partial_json: "{\"location\":".to_string(),
            },
        };
        let result1 = handle_event(&event1, &mut tool_call_state, &mut thinking_state);
        assert!(result1.is_some());

        // Second delta
        let event2 = StreamingEvent::ContentBlockDelta {
            index: 0,
            delta: ContentDelta::InputJsonDelta {
                partial_json: "\"Paris\",".to_string(),
            },
        };
        let result2 = handle_event(&event2, &mut tool_call_state, &mut thinking_state);
        assert!(result2.is_some());

        // Third delta
        let event3 = StreamingEvent::ContentBlockDelta {
            index: 0,
            delta: ContentDelta::InputJsonDelta {
                partial_json: "\"temp\":\"20C\"}".to_string(),
            },
        };
        let result3 = handle_event(&event3, &mut tool_call_state, &mut thinking_state);
        assert!(result3.is_some());

        assert!(tool_call_state.is_some());

        // Final ContentBlockStop hands the block to the shared accumulator,
        // which finalizes the assembled fragments (`Error` policy: a stopped
        // block promised complete input). End-to-end assembly of exactly this
        // fragment sequence is pinned in `streaming::parts` unit tests.
        let stop_event = StreamingEvent::ContentBlockStop { index: 0 };
        let final_result = handle_event(&stop_event, &mut tool_call_state, &mut thinking_state);
        assert!(final_result.is_some());

        match final_result.unwrap().unwrap() {
            RawStreamingChoice::ToolInputEnd(end) => {
                assert_eq!(end.id.as_wire(), Some("tool_123"));
                assert!(matches!(
                    end.on_unparseable,
                    crate::streaming::UnparseableToolInput::Error
                ));
            }
            other => panic!("Expected ToolInputEnd, got {:?}", other),
        }

        // Tool call state should be taken
        assert!(tool_call_state.is_none());
    }

    #[test]
    fn test_citations_delta_streaming_event_deserialization() {
        let json = r#"{
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "citations_delta",
                "citation": {
                    "type": "char_location",
                    "cited_text": "The grass is green.",
                    "document_index": 0,
                    "document_title": "Example",
                    "start_char_index": 0,
                    "end_char_index": 20
                }
            }
        }"#;

        let event: StreamingEvent = serde_json::from_str(json).unwrap();
        let StreamingEvent::ContentBlockDelta { index, delta } = event else {
            panic!("expected ContentBlockDelta");
        };
        assert_eq!(index, 0);
        let ContentDelta::CitationsDelta { citation } = delta else {
            panic!("expected CitationsDelta");
        };
        let crate::providers::anthropic::completion::Citation::CharLocation {
            start_char_index,
            end_char_index,
            ..
        } = citation
        else {
            panic!("expected CharLocation");
        };
        assert_eq!(start_char_index, 0);
        assert_eq!(end_char_index, 20);
    }

    #[test]
    fn test_search_result_citations_delta_streaming_event_deserialization() {
        let json = r#"{
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "citations_delta",
                "citation": {
                    "type": "search_result_location",
                    "cited_text": "API requests require a key.",
                    "source": "https://docs.example.com/api-reference",
                    "title": "API Reference",
                    "search_result_index": 0,
                    "start_block_index": 0,
                    "end_block_index": 1
                }
            }
        }"#;

        let event: StreamingEvent = serde_json::from_str(json).unwrap();
        let StreamingEvent::ContentBlockDelta { delta, .. } = event else {
            panic!("expected ContentBlockDelta");
        };
        let ContentDelta::CitationsDelta { citation } = delta else {
            panic!("expected CitationsDelta");
        };
        assert!(matches!(
            citation,
            crate::providers::anthropic::completion::Citation::SearchResultLocation {
                search_result_index: 0,
                start_block_index: 0,
                end_block_index: 1,
                ..
            }
        ));
    }

    #[test]
    fn test_web_search_result_citations_delta_streaming_event_deserialization() {
        let json = r#"{
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "citations_delta",
                "citation": {
                    "type": "web_search_result_location",
                    "cited_text": "Claude Shannon was a mathematician.",
                    "url": "https://example.com/shannon",
                    "title": "Claude Shannon",
                    "encrypted_index": "encrypted-reference"
                }
            }
        }"#;

        let event: StreamingEvent = serde_json::from_str(json).unwrap();
        let StreamingEvent::ContentBlockDelta { delta, .. } = event else {
            panic!("expected ContentBlockDelta");
        };
        let ContentDelta::CitationsDelta { citation } = delta else {
            panic!("expected CitationsDelta");
        };
        assert!(matches!(
            citation,
            crate::providers::anthropic::completion::Citation::WebSearchResultLocation {
                ref url,
                ref encrypted_index,
                ..
            } if url == "https://example.com/shannon"
                && encrypted_index == "encrypted-reference"
        ));
    }

    #[test]
    fn test_web_search_result_citations_delta_allows_null_title() {
        let json = r#"{
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "citations_delta",
                "citation": {
                    "type": "web_search_result_location",
                    "cited_text": "Claude Shannon was a mathematician.",
                    "url": "https://example.com/shannon",
                    "title": null,
                    "encrypted_index": "encrypted-reference"
                }
            }
        }"#;

        let event: StreamingEvent = serde_json::from_str(json).unwrap();
        let StreamingEvent::ContentBlockDelta { delta, .. } = event else {
            panic!("expected ContentBlockDelta");
        };
        let ContentDelta::CitationsDelta { citation } = delta else {
            panic!("expected CitationsDelta");
        };
        assert!(matches!(
            citation,
            crate::providers::anthropic::completion::Citation::WebSearchResultLocation {
                title: None,
                ..
            }
        ));
    }

    #[test]
    fn test_text_content_block_start_allows_null_citations() {
        // The Anthropic Messages API emits an explicit `"citations": null` on the
        // first text `content_block_start` event. `#[serde(default)]` alone covers
        // a missing field but not an explicit null, so this must deserialize to an
        // empty citation list rather than failing the whole stream (see #1971).
        let json = r#"{
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "text",
                "text": "",
                "citations": null
            }
        }"#;

        let event: StreamingEvent = serde_json::from_str(json).unwrap();
        let StreamingEvent::ContentBlockStart { content_block, .. } = event else {
            panic!("expected ContentBlockStart");
        };
        let Content::Text {
            text, citations, ..
        } = content_block
        else {
            panic!("expected text content block");
        };
        assert_eq!(text, "");
        assert!(citations.is_empty());
    }

    #[test]
    fn test_web_search_content_block_start_events_deserialize() {
        let server_tool_use = r#"{
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "server_tool_use",
                "id": "srvtoolu_01",
                "name": "web_search",
                "input": {
                    "query": "claude shannon birth date"
                }
            }
        }"#;
        let event: StreamingEvent = serde_json::from_str(server_tool_use).unwrap();
        assert!(matches!(
            event,
            StreamingEvent::ContentBlockStart {
                content_block: Content::ServerToolUse {
                    ref id,
                    ref name,
                    ref input
                },
                ..
            } if id == "srvtoolu_01"
                && name == "web_search"
                && input["query"] == "claude shannon birth date"
        ));

        let web_search_tool_result = r#"{
            "type": "content_block_start",
            "index": 2,
            "content_block": {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_01",
                "content": [{
                    "type": "web_search_result",
                    "url": "https://example.com/shannon",
                    "title": "Claude Shannon",
                    "encrypted_content": "encrypted-content"
                }]
            }
        }"#;
        let event: StreamingEvent = serde_json::from_str(web_search_tool_result).unwrap();
        assert!(matches!(
            event,
            StreamingEvent::ContentBlockStart {
                content_block: Content::WebSearchToolResult {
                    ref tool_use_id,
                    ref content
                },
                ..
            } if tool_use_id == "srvtoolu_01"
                && content[0]["encrypted_content"] == "encrypted-content"
        ));
    }

    #[test]
    fn test_code_execution_tool_result_block_is_preserved() {
        let event: StreamingEvent = serde_json::from_value(serde_json::json!({
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "code_execution_tool_result",
                "tool_use_id": "srvtoolu_01",
                "content": {
                    "type": "code_execution_result",
                    "return_code": 0,
                    "stdout": "42\n",
                    "stderr": "",
                    "content": []
                }
            }
        }))
        .unwrap();
        let mut tool_call_state = None;
        let mut server_tool_uses = HashMap::new();
        let mut thinking_state = None;

        let choice = super::handle_event(
            &event,
            &mut tool_call_state,
            &mut server_tool_uses,
            &mut thinking_state,
        )
        .expect("code_execution_tool_result block should produce raw metadata")
        .unwrap();

        let RawStreamingChoice::TextStart {
            id,
            additional_params: Some(additional_params),
        } = choice
        else {
            panic!("expected text-start metadata for code_execution_tool_result");
        };
        assert_eq!(id, crate::streaming::MintKind::Block.for_wire_index(1));
        assert_eq!(
            additional_params[crate::providers::anthropic::completion::ANTHROPIC_RAW_CONTENT_KEY]["type"],
            "code_execution_tool_result"
        );
        assert_eq!(
            additional_params[crate::providers::anthropic::completion::ANTHROPIC_RAW_CONTENT_KEY]["content"]
                ["stdout"],
            "42\n"
        );
    }

    #[tokio::test]
    async fn test_streaming_web_search_blocks_are_preserved_on_final_choice() {
        let raw_stream = stream! {
            let mut tool_call_state = None;
            let mut server_tool_uses = HashMap::new();
            let mut thinking_state = None;

            let server_tool_use_start = super::handle_event(
                &StreamingEvent::ContentBlockStart {
                    index: 0,
                    content_block: Content::ServerToolUse {
                        id: "srvtoolu_01".to_string(),
                        name: "web_search".to_string(),
                        input: serde_json::Value::Null,
                    },
                },
                &mut tool_call_state,
                &mut server_tool_uses,
                &mut thinking_state,
            );
            assert!(
                server_tool_use_start.is_none(),
                "server_tool_use start should be accumulated until its input JSON is complete"
            );

            let server_tool_use_delta = super::handle_event(
                &StreamingEvent::ContentBlockDelta {
                    index: 0,
                    delta: ContentDelta::InputJsonDelta {
                        partial_json: r#"{"query":"claude shannon birth date"}"#.to_string(),
                    },
                },
                &mut tool_call_state,
                &mut server_tool_uses,
                &mut thinking_state,
            );
            assert!(
                server_tool_use_delta.is_none(),
                "server_tool_use input JSON should not be emitted as a Rig tool-call delta"
            );

            yield super::handle_event(
                &StreamingEvent::ContentBlockStop { index: 0 },
                &mut tool_call_state,
                &mut server_tool_uses,
                &mut thinking_state,
            )
            .expect("server_tool_use stop should produce completed raw metadata");

            yield super::handle_event(
                &StreamingEvent::ContentBlockStart {
                    index: 1,
                    content_block: Content::WebSearchToolResult {
                        tool_use_id: "srvtoolu_01".to_string(),
                        content: serde_json::json!([{
                            "type": "web_search_result",
                            "url": "https://example.com/shannon",
                            "title": "Claude Shannon",
                            "encrypted_content": "encrypted-content"
                        }]),
                    },
                },
                &mut tool_call_state,
                &mut server_tool_uses,
                &mut thinking_state,
            )
            .expect("web_search_tool_result block should produce raw metadata");

            yield super::handle_event(
                &StreamingEvent::ContentBlockStart {
                    index: 2,
                    content_block: Content::Text {
                        text: String::new(),
                        citations: Vec::new(),
                        cache_control: None,
                    },
                },
                &mut tool_call_state,
                &mut server_tool_uses,
                &mut thinking_state,
            )
            .expect("text block start should produce a raw choice");

            yield super::handle_event(
                &StreamingEvent::ContentBlockDelta {
                    index: 2,
                    delta: ContentDelta::TextDelta {
                        text: "Claude Shannon was born on April 30, 1916.".to_string(),
                    },
                },
                &mut tool_call_state,
                &mut server_tool_uses,
                &mut thinking_state,
            )
            .expect("text delta should produce a raw choice");

            yield super::handle_event(
                &StreamingEvent::ContentBlockDelta {
                    index: 2,
                    delta: ContentDelta::CitationsDelta {
                        citation: crate::providers::anthropic::completion::Citation::WebSearchResultLocation {
                            cited_text: "Claude Shannon was born on April 30, 1916.".to_string(),
                            url: "https://example.com/shannon".to_string(),
                            title: Some("Claude Shannon".to_string()),
                            encrypted_index: "encrypted-index".to_string(),
                        },
                    },
                },
                &mut tool_call_state,
                &mut server_tool_uses,
                &mut thinking_state,
            )
            .expect("citation delta should produce a raw choice");

            yield Ok(RawStreamingChoice::FinalResponse(StreamingCompletionResponse::default()));
        };

        let mut stream = crate::streaming::StreamingCompletionResponse::stream(
            "anthropic",
            to_stream_result(raw_stream),
        );
        while stream.next().await.is_some() {}

        let choice_items: Vec<crate::message::AssistantContent> =
            stream.choice.clone().into_iter().collect();
        assert_eq!(choice_items.len(), 3);
        assert!(
            choice_items
                .iter()
                .all(|item| !matches!(item, crate::message::AssistantContent::ToolCall(_))),
            "provider-owned web-search blocks must not become Rig client tool calls"
        );

        let Some(crate::message::AssistantContent::Text(server_tool_use)) = choice_items.first()
        else {
            panic!("expected raw server_tool_use metadata");
        };
        assert_eq!(
            server_tool_use.additional_params.as_ref().unwrap()
                [crate::providers::anthropic::completion::ANTHROPIC_RAW_CONTENT_KEY]["type"],
            "server_tool_use"
        );
        assert_eq!(
            server_tool_use.additional_params.as_ref().unwrap()
                [crate::providers::anthropic::completion::ANTHROPIC_RAW_CONTENT_KEY]["input"]["query"],
            "claude shannon birth date"
        );

        let Some(crate::message::AssistantContent::Text(web_search_result)) = choice_items.get(1)
        else {
            panic!("expected raw web_search_tool_result metadata");
        };
        assert_eq!(
            web_search_result.additional_params.as_ref().unwrap()
                [crate::providers::anthropic::completion::ANTHROPIC_RAW_CONTENT_KEY]["content"][0]
                ["encrypted_content"],
            "encrypted-content"
        );

        let Some(crate::message::AssistantContent::Text(answer)) = choice_items.get(2) else {
            panic!("expected answer text");
        };
        assert_eq!(answer.text, "Claude Shannon was born on April 30, 1916.");
        let citations = crate::providers::anthropic::completion::anthropic_citations(answer)
            .expect("expected preserved citations");
        assert!(matches!(
            citations.first(),
            Some(crate::providers::anthropic::completion::Citation::WebSearchResultLocation {
                encrypted_index,
                ..
            }) if encrypted_index == "encrypted-index"
        ));
    }

    #[test]
    fn test_handle_citations_delta_event_preserves_metadata() {
        let event = StreamingEvent::ContentBlockDelta {
            index: 0,
            delta: ContentDelta::CitationsDelta {
                citation: crate::providers::anthropic::completion::Citation::CharLocation {
                    cited_text: "The grass is green.".to_string(),
                    document_index: 0,
                    document_title: Some("Example".to_string()),
                    start_char_index: 0,
                    end_char_index: 20,
                },
            },
        };

        let mut tool_call_state = None;
        let mut thinking_state = None;
        let result = handle_event(&event, &mut tool_call_state, &mut thinking_state);

        assert!(result.is_some());
        let choice = result.unwrap().unwrap();
        let RawStreamingChoice::TextAdditionalParams(additional_params) = choice else {
            panic!("expected TextAdditionalParams choice");
        };
        assert_eq!(additional_params["citations"][0]["type"], "char_location");
    }

    #[tokio::test]
    async fn test_streaming_citation_deltas_are_preserved_on_final_text() {
        let citation = crate::providers::anthropic::completion::Citation::CharLocation {
            cited_text: "The grass is green.".to_string(),
            document_index: 0,
            document_title: Some("Example".to_string()),
            start_char_index: 0,
            end_char_index: 20,
        };

        let raw_stream = stream! {
            let mut tool_call_state = None;
            let mut thinking_state = None;

            yield handle_event(
                &StreamingEvent::ContentBlockStart {
                    index: 0,
                    content_block: Content::Text {
                        text: String::new(),
                        citations: Vec::new(),
                        cache_control: None,
                    },
                },
                &mut tool_call_state,
                &mut thinking_state,
            )
            .expect("text block start should produce a raw choice");

            yield handle_event(
                &StreamingEvent::ContentBlockDelta {
                    index: 0,
                    delta: ContentDelta::TextDelta {
                        text: "the grass is green".to_string(),
                    },
                },
                &mut tool_call_state,
                &mut thinking_state,
            )
            .expect("text delta should produce a raw choice");

            yield handle_event(
                &StreamingEvent::ContentBlockDelta {
                    index: 0,
                    delta: ContentDelta::CitationsDelta {
                        citation: crate::providers::anthropic::completion::Citation::CharLocation {
                            cited_text: "The grass is green.".to_string(),
                            document_index: 0,
                            document_title: Some("Example".to_string()),
                            start_char_index: 0,
                            end_char_index: 20,
                        },
                    },
                },
                &mut tool_call_state,
                &mut thinking_state,
            )
            .expect("citation delta should produce a raw choice");

            yield Ok(RawStreamingChoice::FinalResponse(StreamingCompletionResponse::default()));
        };

        let mut stream = crate::streaming::StreamingCompletionResponse::stream(
            "anthropic",
            to_stream_result(raw_stream),
        );
        while stream.next().await.is_some() {}

        let choice_items: Vec<crate::message::AssistantContent> =
            stream.choice.clone().into_iter().collect();
        let Some(crate::message::AssistantContent::Text(text)) = choice_items.first() else {
            panic!("expected accumulated text item");
        };

        assert_eq!(text.text, "the grass is green");
        let citations = crate::providers::anthropic::completion::anthropic_citations(text).unwrap();
        assert_eq!(citations, vec![citation]);
    }

    /// The `#[serde(other)]` policy fallbacks are gone: classification is the
    /// only policy site. An unmodeled *top-level* event type is `Unknown`
    /// (driver: warn + skip); a `ping` is Known; and a known tag whose payload
    /// this client cannot decode is `Corrupt`, never silently demoted to an
    /// ignorable unknown. An unmodeled *nested* delta type is the one carved
    /// exception (Anthropic's versioning policy reserves the right to add
    /// them): it decodes to [`ContentDelta::Unknown`] and stays a Known
    /// no-op — see the dedicated tests below.
    #[test]
    fn classify_dispatches_on_the_known_event_list() {
        let adapter = AnthropicAdapter::default();

        let frame =
            WireFrame::Text(r#"{"type":"something_new_from_anthropic","field":"x"}"#.into());
        assert!(matches!(
            adapter.classify(frame),
            crate::providers::internal::wire::WireEvent::Unknown { event_type, .. }
                if event_type == "something_new_from_anthropic"
        ));

        let frame = WireFrame::Text(r#"{"type":"ping"}"#.into());
        assert!(matches!(
            adapter.classify(frame),
            crate::providers::internal::wire::WireEvent::Known(StreamingEvent::Ping)
        ));

        let frame = WireFrame::Text("{not json".into());
        assert!(matches!(
            adapter.classify(frame),
            crate::providers::internal::wire::WireEvent::Corrupt(_)
        ));
    }

    /// Forward compat: a novel nested delta type Anthropic ships tomorrow
    /// must not corrupt the whole `content_block_delta` frame — it decodes
    /// to [`ContentDelta::Unknown`] and interprets as a warned no-op, so the
    /// stream continues.
    #[test]
    fn novel_nested_delta_type_is_a_known_noop() {
        let adapter = AnthropicAdapter::default();
        let frame = WireFrame::Text(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"banana_delta","x":1}}"#
                .into(),
        );
        let crate::providers::internal::wire::WireEvent::Known(event) = adapter.classify(frame)
        else {
            panic!("a novel nested delta type must stay a Known event");
        };

        let mut adapter = AnthropicAdapter::default();
        let mut out = Vec::new();
        adapter.interpret(event, &mut out);
        assert!(out.is_empty(), "an unmodeled nested delta is a no-op");
    }

    /// A `content_block_delta` whose `delta` omits `type` is malformed, not
    /// novel: silently skipping it would turn a compat gateway's untagged
    /// text delta into a successful *empty* completion. It classifies
    /// `Corrupt`, surfacing in-band while the stream keeps consuming
    /// (#2258 B5).
    #[test]
    fn delta_missing_its_type_is_corrupt_not_skipped() {
        let adapter = AnthropicAdapter::default();
        let frame = WireFrame::Text(
            r#"{"type":"content_block_delta","index":0,"delta":{"text":"hello"}}"#.into(),
        );
        assert!(matches!(
            adapter.classify(frame),
            crate::providers::internal::wire::WireEvent::Corrupt(_)
        ));
    }

    /// Policy preserved: a *known* nested delta tag with a defective payload
    /// is a data-level defect, not an unmodeled delta — the frame classifies
    /// `Corrupt` instead of degrading to an `Unknown` no-op.
    #[test]
    fn known_nested_delta_tag_with_defective_payload_is_corrupt() {
        let adapter = AnthropicAdapter::default();
        let frame = WireFrame::Text(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":42}}"#
                .into(),
        );
        assert!(matches!(
            adapter.classify(frame),
            crate::providers::internal::wire::WireEvent::Corrupt(_)
        ));
    }

    /// Anthropic's top-level `{"type":"error"}` envelope (e.g.
    /// `overloaded_error`) is a Known event that surfaces as a provider error
    /// carrying the full envelope — never a warn-skipped unknown — and, since
    /// no `message_delta` follows, the stream ends with no terminal record.
    #[test]
    fn top_level_error_event_surfaces_as_a_provider_error() {
        let adapter = AnthropicAdapter::default();
        let frame = WireFrame::Text(
            r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#.into(),
        );
        let crate::providers::internal::wire::WireEvent::Known(event) = adapter.classify(frame)
        else {
            panic!("the error envelope must classify as a Known event");
        };

        let mut adapter = AnthropicAdapter::default();
        let mut out = Vec::new();
        adapter.interpret(event, &mut out);

        assert_eq!(out.len(), 1, "the error envelope maps to one error item");
        let Some(Err(error)) = out.pop() else {
            panic!("the error envelope must surface as an Err item");
        };
        let body = error
            .provider_response_body()
            .expect("the provider's error payload must be preserved");
        assert!(
            body.contains("overloaded_error") && body.contains("Overloaded"),
            "the full envelope must survive into the error body, got: {body}"
        );
    }

    /// Bedrock-compat quirk: `message_start` without a message body is a
    /// Known no-op, not a corrupt frame.
    #[test]
    fn message_start_with_null_message_is_a_known_noop() {
        let adapter = AnthropicAdapter::default();
        let frame = WireFrame::Text(r#"{"type":"message_start","message":null}"#.into());
        let crate::providers::internal::wire::WireEvent::Known(event) = adapter.classify(frame)
        else {
            panic!("null-message message_start must stay a known event");
        };

        let mut adapter = AnthropicAdapter::default();
        let mut out = Vec::new();
        adapter.interpret(event, &mut out);
        assert!(out.is_empty(), "a message-less message_start is a no-op");
    }

    #[tokio::test]
    async fn terminal_record_normalizes_stop_reason_usage_and_metadata() {
        let raw_stream = stream! {
            yield Ok(RawStreamingChoice::Message("hi".to_string()));
            yield Ok(RawStreamingChoice::FinalResponse(StreamingCompletionResponse {
                usage: PartialUsage {
                    output_tokens: 5,
                    input_tokens: Some(3),
                    cache_creation_input_tokens: None,
                    cache_read_input_tokens: Some(2),
                },
                stop_reason: Some("max_tokens".to_string()),
                message_id: Some("msg_1".to_string()),
                model: Some(CLAUDE_OPUS_4_8.to_string()),
            }));
        };

        let mut stream = crate::streaming::StreamingCompletionResponse::stream(
            "anthropic",
            to_stream_result(raw_stream),
        );
        while stream.next().await.is_some() {}

        let terminal = stream.response.expect("expected a terminal record");
        assert_eq!(terminal.provider, "anthropic");
        assert_eq!(terminal.message_id.as_deref(), Some("msg_1"));
        assert_eq!(terminal.model.as_deref(), Some(CLAUDE_OPUS_4_8));
        assert_eq!(
            terminal.finish_reason,
            Some(crate::completion::FinishReason::Length)
        );
        assert_eq!(terminal.usage.input_tokens, 3);
        assert_eq!(terminal.usage.output_tokens, 5);
        assert_eq!(terminal.usage.cached_input_tokens, 2);
        assert_eq!(terminal.usage.total_tokens, 10);
    }

    #[tokio::test]
    async fn terminal_record_upgrades_end_turn_to_tool_calls_after_a_streamed_tool_call() {
        // Anthropic normally reports `tool_use`, but the reconciliation
        // `normalize_stream` applies must hold whenever the turn actually
        // emitted a tool call.
        let raw_stream = stream! {
            yield Ok(RawStreamingChoice::ToolCall(RawStreamingToolCall::new(
                "toolu_1".to_string(),
                "add".to_string(),
                json!({"x": 1}),
            )));
            yield Ok(RawStreamingChoice::FinalResponse(StreamingCompletionResponse {
                stop_reason: Some("end_turn".to_string()),
                ..Default::default()
            }));
        };

        let mut stream = crate::streaming::StreamingCompletionResponse::stream(
            "anthropic",
            to_stream_result(raw_stream),
        );
        while stream.next().await.is_some() {}

        let terminal = stream.response.expect("expected a terminal record");
        assert_eq!(
            terminal.finish_reason,
            Some(crate::completion::FinishReason::ToolCalls)
        );
    }

    #[tokio::test]
    async fn unknown_stop_reason_survives_onto_the_terminal_record() {
        let raw_stream = stream! {
            yield Ok(RawStreamingChoice::FinalResponse(StreamingCompletionResponse {
                stop_reason: Some("pause_turn".to_string()),
                ..Default::default()
            }));
        };

        let mut stream = crate::streaming::StreamingCompletionResponse::stream(
            "anthropic",
            to_stream_result(raw_stream),
        );
        while stream.next().await.is_some() {}

        let terminal = stream.response.expect("expected a terminal record");
        assert_eq!(
            terminal.finish_reason,
            Some(crate::completion::FinishReason::Other(
                "pause_turn".to_owned()
            ))
        );
    }

    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    mod terminal_emission {
        use super::super::super::completion::CLAUDE_SONNET_4_6;
        use crate::client::CompletionClient;
        use crate::completion::CompletionModel as _;
        use crate::providers::anthropic::Client;
        use crate::streaming::StreamedAssistantContent;
        use crate::test_utils::MockStreamingClient;
        use futures::StreamExt;

        const MESSAGE_START: &str = r#"{"type":"message_start","message":{"id":"msg_1","role":"assistant","content":[],"model":"claude-sonnet-4-6","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":5,"output_tokens":0}}}"#;
        const TEXT_START: &str =
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#;
        const TEXT_DELTA: &str =
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}"#;
        const MESSAGE_DELTA: &str = r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":3}}"#;

        fn sse(frames: &[&str]) -> bytes::Bytes {
            bytes::Bytes::from(
                frames
                    .iter()
                    .map(|frame| format!("data: {frame}\n\n"))
                    .collect::<String>(),
            )
        }

        async fn collect(
            sse_bytes: bytes::Bytes,
        ) -> (
            Vec<String>,
            bool,
            bool,
            crate::streaming::StreamingCompletionResponse,
        ) {
            let client = Client::builder()
                .api_key("test-key")
                .http_client(MockStreamingClient { sse_bytes })
                .build()
                .expect("build client");
            let model = client.completion_model(CLAUDE_SONNET_4_6);
            let request = model.completion_request("hello").build();
            let mut stream = crate::completion::CompletionModel::stream(&model, request)
                .await
                .expect("stream should open");

            let mut texts = Vec::new();
            let mut saw_error = false;
            let mut saw_terminal = false;
            while let Some(item) = stream.next().await {
                match item {
                    Ok(StreamedAssistantContent::Text(text)) => texts.push(text.text),
                    Ok(StreamedAssistantContent::Final(_)) => saw_terminal = true,
                    Ok(_) => {}
                    Err(_) => saw_error = true,
                }
            }
            (texts, saw_error, saw_terminal, stream)
        }

        #[tokio::test]
        async fn truncated_stream_yields_content_but_no_terminal_record() {
            let (texts, saw_error, saw_terminal, stream) =
                collect(sse(&[MESSAGE_START, TEXT_START, TEXT_DELTA])).await;

            assert_eq!(texts, ["hi"]);
            assert!(!saw_error);
            assert!(
                !saw_terminal,
                "EOF without message_delta must not synthesize a terminal record"
            );
            assert!(stream.response.is_none());
        }

        #[tokio::test]
        async fn errored_stream_forwards_the_error_and_no_terminal_record() {
            use crate::test_utils::SequencedStreamingHttpClient;

            // A transport failure injected into the byte stream after some
            // content must be forwarded (via `from_stream_transport`) and must
            // not be papered over with a synthesized terminal record.
            let client = Client::builder()
                .api_key("test-key")
                .http_client(SequencedStreamingHttpClient::new(vec![
                    Ok(sse(&[MESSAGE_START, TEXT_START, TEXT_DELTA])),
                    Err(crate::http_client::Error::InvalidStatusCodeWithMessage(
                        http::StatusCode::BAD_GATEWAY,
                        "connection reset".to_string(),
                    )),
                ]))
                .build()
                .expect("build client");
            let model = client.completion_model(CLAUDE_SONNET_4_6);
            let request = model.completion_request("hello").build();
            let mut stream = crate::completion::CompletionModel::stream(&model, request)
                .await
                .expect("stream should open");

            let mut texts = Vec::new();
            let mut saw_error = false;
            let mut saw_terminal = false;
            while let Some(item) = stream.next().await {
                match item {
                    Ok(StreamedAssistantContent::Text(text)) => texts.push(text.text),
                    Ok(StreamedAssistantContent::Final(_)) => saw_terminal = true,
                    Ok(_) => {}
                    Err(_) => saw_error = true,
                }
            }

            assert_eq!(texts, ["hi"]);
            assert!(saw_error, "the transport failure must reach the consumer");
            assert!(
                !saw_terminal,
                "a failed stream must not synthesize a terminal record"
            );
            assert!(stream.response.is_none());
        }

        #[tokio::test]
        async fn provider_error_event_stops_the_stream_before_a_later_terminal() {
            // The findings-file probe: an in-band provider `error` event
            // followed by a well-formed `message_delta`. The error must reach
            // the consumer and NOTHING may follow it — the adapter is
            // finished, so the later terminal frame must not be interpreted
            // into a successful FinalResponse.
            const ERROR_EVENT: &str =
                r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#;
            let (texts, saw_error, saw_terminal, stream) = collect(sse(&[
                MESSAGE_START,
                TEXT_START,
                TEXT_DELTA,
                ERROR_EVENT,
                MESSAGE_DELTA,
            ]))
            .await;

            assert_eq!(texts, ["hi"]);
            assert!(saw_error, "the provider error must reach the consumer");
            assert!(
                !saw_terminal,
                "a message_delta after an in-band provider error must not read as a completed turn"
            );
            assert!(stream.response.is_none());
        }

        #[tokio::test]
        async fn malformed_frame_then_eof_yields_error_and_no_terminal_record() {
            let (texts, saw_error, saw_terminal, stream) =
                collect(sse(&[MESSAGE_START, TEXT_START, TEXT_DELTA, "{not json"])).await;

            assert_eq!(texts, ["hi"]);
            assert!(saw_error, "the malformed frame must reach the consumer");
            assert!(
                !saw_terminal,
                "a parse error followed by EOF must not read as a completed turn"
            );
            assert!(stream.response.is_none());
        }

        #[tokio::test]
        async fn malformed_frame_then_real_terminal_still_completes_the_stream() {
            let (texts, saw_error, saw_terminal, stream) = collect(sse(&[
                MESSAGE_START,
                TEXT_START,
                TEXT_DELTA,
                "{not json",
                MESSAGE_DELTA,
            ]))
            .await;

            assert_eq!(texts, ["hi"]);
            assert!(saw_error, "the malformed frame must reach the consumer");
            assert!(
                saw_terminal,
                "a genuine message_delta after a parse error still completes the stream"
            );
            let terminal = stream.response.expect("terminal record");
            assert_eq!(
                terminal.finish_reason,
                Some(crate::completion::FinishReason::Stop)
            );
            assert_eq!(terminal.message_id.as_deref(), Some("msg_1"));
        }
    }
}
