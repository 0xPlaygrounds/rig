use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::completion::{CompletionResponse, Content, anthropic_usage_totals, map_finish_reason};
use crate::error::ProviderError;
use crate::message::ReasoningContent;
use crate::observe::ObservedError;
use crate::operation::{AdapterOutput, Completion};
use crate::providers::internal::wire::{self, WireEvent};
use crate::streaming::{self, BlockId, MintKind, StreamFinal, ToolCallEnd, UnparseableToolInput};
use crate::wire::{
    AdapterEvent, AdapterUsage, AdapterVerdict, Decoder, ObservationSink, WireFrame,
};
use std::collections::HashMap;

/// Recognized Messages event tags. Listed events must decode fully;
/// unlisted tags classify as unknown. Novel nested delta tags remain
/// [`ContentDelta::Unknown`].
const KNOWN_EVENT_TYPES: &[&str] = &[
    // Unary replies use the same classifier with a whole-message tag.
    "message",
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
        /// Initial message metadata. Absent or null messages are accepted as no-ops.
        #[serde(default)]
        message: Option<CompletionResponse>,
    },
    /// The whole message: what the endpoint answers when not streaming.
    /// Its fields are exactly `message_start`'s, plus the stop reason and
    /// usage a stream delivers on `message_delta`.
    Message {
        #[serde(flatten)]
        message: CompletionResponse,
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
    /// A provider error envelope with a required nested `error` field.
    Error {
        /// Required error payload used to validate the envelope shape.
        #[allow(dead_code)]
        error: serde_json::Value,
        /// Original envelope bytes attached during classification.
        /// Preserve sibling fields and key order in the reported provider error.
        #[serde(skip)]
        raw: String,
    },
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
    /// An unrecognized nested delta tag, preserved for a warning and skipped.
    Unknown(serde_json::Value),
}

/// Decode known delta tags strictly and preserve unrecognized string tags.
/// Reject non-object values, missing or non-string tags, and malformed known payloads.
impl<'de> Deserialize<'de> for ContentDelta {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        // Non-object values are malformed, not novel delta kinds.
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
            // Missing tags must not silently turn discarded content into successful output.
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
    /// Output-token breakdown reported by the terminal `message_delta`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<super::completion::OutputTokensDetails>,
}

impl From<&PartialUsage> for crate::completion::Usage {
    fn from(value: &PartialUsage) -> crate::completion::Usage {
        anthropic_usage_totals(
            value.input_tokens.map(|tokens| tokens as u64),
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

// Hosted-tool input is assembled locally because it becomes raw text-block
// metadata rather than an executable tool call.
struct ServerToolUseState {
    name: String,
    id: String,
    initial_input: Value,
    input_json: String,
}

#[derive(Default)]
struct ThinkingState {
    /// Signature fragments assembled for the block-end event.
    signature: String,
    /// Opening signature, used only when no nonempty signature deltas arrive.
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

/// Decode unary and streamed Messages replies into canonical content and terminal events.
pub struct MessagesDecoder {
    /// Selected dialect's provider name for terminal records.
    provider: &'static str,
    /// Wire id of the open client tool-use block, when one is streaming.
    current_tool_call: Option<BlockId>,
    /// Keys for calls whose wire id is empty: an absent id is not an id.
    tool_ids: streaming::SyntheticIds,
    server_tool_uses: HashMap<usize, ServerToolUseState>,
    current_thinking: Option<ThinkingState>,
    input_tokens: u64,
    /// Per-TTL cache-write breakdown from `message_start`; the terminal
    /// `message_delta` usage omits it.
    cache_creation: Option<super::completion::CacheCreation>,
    message_id: Option<String>,
    response_model: Option<String>,
    /// A terminal error was emitted; subsequent frames must not produce output.
    failed: bool,
}

impl MessagesDecoder {
    /// A fresh decoder whose terminal record names `provider`.
    pub fn new(provider: &'static str) -> Self {
        Self {
            provider,
            current_tool_call: None,
            tool_ids: streaming::SyntheticIds::tool(),
            server_tool_uses: HashMap::new(),
            current_thinking: None,
            input_tokens: 0,
            cache_creation: None,
            message_id: None,
            response_model: None,
            failed: false,
        }
    }

    /// The content-block frames: `content_block_start` / `_delta` / `_stop`.
    fn interpret_content(&mut self, event: StreamingEvent, out: &mut AdapterOutput) {
        match event {
            StreamingEvent::ContentBlockDelta { index, delta } => match delta {
                ContentDelta::TextDelta { text } => {
                    if self.current_tool_call.is_none() {
                        out.text(text);
                    }
                }
                ContentDelta::InputJsonDelta { partial_json } => {
                    if let Some(server_tool_use) = self.server_tool_uses.get_mut(&index) {
                        server_tool_use.input_json.push_str(&partial_json);
                        return;
                    }

                    if let Some(key) = &self.current_tool_call {
                        // Emit the delta so UI can show progress; the shared
                        // accumulator assembles the fragments.
                        out.tool_arguments(key, partial_json);
                    }
                }
                ContentDelta::ThinkingDelta { thinking } => {
                    self.current_thinking
                        .get_or_insert_with(ThinkingState::default);

                    // Anthropic has no reasoning item id; the content-block
                    // index is stable across a block's deltas and its stop.
                    out.reasoning_delta(
                        &MintKind::Block.for_wire_index(index as u64),
                        None,
                        thinking,
                    );
                }
                ContentDelta::SignatureDelta { signature } => {
                    self.current_thinking
                        .get_or_insert_with(ThinkingState::default)
                        .signature
                        .push_str(&signature);

                    // The completed signature belongs on the thinking block's end event.
                }
                ContentDelta::CitationsDelta { citation } => {
                    if let Some(params) = crate::message::AdditionalParams::from_entries([(
                        "citations",
                        json!([citation]),
                    )]) {
                        out.text_meta(params);
                    }
                }
                ContentDelta::Unknown(value) => {
                    // Log only the tag; unknown payloads may contain sensitive model output.
                    tracing::warn!(
                        delta_type = value.get("type").and_then(serde_json::Value::as_str),
                        "skipping unrecognized Anthropic content delta type"
                    );
                }
            },
            StreamingEvent::ContentBlockStart {
                index,
                content_block,
            } => match content_block {
                // Text arrives through deltas; cache_control is request-only metadata.
                Content::Text {
                    text: _,
                    citations,
                    cache_control: _,
                } => {
                    let additional_params = crate::message::AdditionalParams::from_entries(
                        (!citations.is_empty()).then(|| ("citations", json!(citations))),
                    );
                    // Anthropic has no text item id; the content-block index
                    // is stable for the block's lifetime.
                    out.text_start(
                        MintKind::Block.for_wire_index(index as u64),
                        additional_params,
                    );
                }
                Content::ServerToolUse { id, name, input } => {
                    self.server_tool_uses.insert(
                        index,
                        ServerToolUseState {
                            name,
                            id,
                            initial_input: input,
                            input_json: String::new(),
                        },
                    );
                }
                raw @ (Content::WebSearchToolResult { .. }
                | Content::CodeExecutionToolResult { .. }) => out.text_start(
                    MintKind::Block.for_wire_index(index as u64),
                    crate::message::AdditionalParams::from_entries([(
                        super::completion::ANTHROPIC_RAW_CONTENT_KEY,
                        json!(raw),
                    )]),
                ),
                Content::ToolUse { id, name, .. } => {
                    let key = streaming::non_empty_id(id)
                        .map_or_else(|| self.tool_ids.mint(), BlockId::wire);
                    self.current_tool_call = Some(key.clone());
                    out.tool_name(&key, name);
                }
                Content::Thinking {
                    thinking,
                    signature,
                } => {
                    // Adaptive thinking may contain only a signature, so retain state
                    // even when the opening text is empty.
                    self.current_thinking = Some(ThinkingState {
                        signature: String::new(),
                        initial_signature: signature.unwrap_or_default(),
                    });
                    // The opening payload's text is a delta like any other;
                    // the shared accumulator owns the block's text.
                    if !thinking.is_empty() {
                        out.reasoning_delta(
                            &MintKind::Block.for_wire_index(index as u64),
                            None,
                            thinking,
                        );
                    }
                }
                Content::RedactedThinking { data } => out.reasoning_block(
                    // Derive the key from the content-block index (no wire id).
                    MintKind::Block.for_wire_index(index as u64),
                    None,
                    ReasoningContent::Redacted { data },
                ),
                // Request-side content kinds; an assistant stream never
                // opens a block with them, and there is nothing to emit.
                Content::Image { .. } | Content::ToolResult { .. } | Content::Document { .. } => {}
            },
            StreamingEvent::ContentBlockStop { index } => {
                // Signature-only thinking blocks carry provider state required for replay.
                if let Some(thinking_state) = self.current_thinking.take() {
                    out.reasoning_end(
                        MintKind::Block.for_wire_index(index as u64),
                        None,
                        thinking_state.into_signature(),
                        // `content_block_stop` is the wire's own end frame,
                        // so even an unsigned block yields its end event.
                        true,
                    );
                    return;
                }

                if let Some(server_tool_use) = self.server_tool_uses.remove(&index) {
                    let input = if server_tool_use.input_json.is_empty() {
                        if server_tool_use.initial_input.is_null() {
                            json!({})
                        } else {
                            server_tool_use.initial_input
                        }
                    } else {
                        match serde_json::from_str(&server_tool_use.input_json) {
                            Ok(json_value) => json_value,
                            Err(e) => {
                                out.error(ProviderError::from(e));
                                return;
                            }
                        }
                    };

                    out.text_start(
                        MintKind::Block.for_wire_index(index as u64),
                        crate::message::AdditionalParams::from_entries([(
                            super::completion::ANTHROPIC_RAW_CONTENT_KEY,
                            json!(Content::ServerToolUse {
                                id: server_tool_use.id,
                                name: server_tool_use.name,
                                input,
                            }),
                        )]),
                    );
                    return;
                }

                // `content_block_stop` promises a complete block: empty input
                // finalizes to `{}`, malformed input surfaces as an error
                // item (`UnparseableToolInput::Error`) in the accumulator.
                if let Some(key) = self.current_tool_call.take() {
                    out.tool_end(key, ToolCallEnd::new(UnparseableToolInput::Error));
                }
            }
            StreamingEvent::Message { .. }
            | StreamingEvent::MessageStart { .. }
            | StreamingEvent::MessageDelta { .. }
            | StreamingEvent::MessageStop
            | StreamingEvent::Ping
            | StreamingEvent::Error { .. } => {}
        }
    }

    /// Interpret a unary reply through the shared content-block lifecycle.
    /// Emit a terminal record after all blocks. Reject empty content unless the
    /// stop reason is `end_turn` or `stop_sequence` with a reported sequence.
    fn interpret_whole_message(&mut self, message: CompletionResponse, out: &mut AdapterOutput) {
        self.input_tokens = message.usage.input_tokens;
        self.cache_creation
            .clone_from(&message.usage.cache_creation);
        self.message_id = Some(message.id);
        self.response_model = Some(message.model);

        // Empty end_turn and stripped stop-sequence replies are valid unary answers.
        // Keep this rejection unary-only; streamed empty turns may complete without error.
        let legal_empty_turn = match message.stop_reason.as_deref() {
            Some("end_turn") => true,
            Some("stop_sequence") => message.stop_sequence.is_some(),
            _ => false,
        };
        if message.content.is_empty() && !legal_empty_turn {
            self.failed = true;
            out.error(ProviderError::Response(
                crate::message::EMPTY_RESPONSE_ERROR.to_owned(),
            ));
            return;
        }

        for (index, content) in message.content.into_iter().enumerate() {
            // The payload a stream delivers by delta, for the part kinds
            // that have one. Everything else is carried by the block's
            // start frame alone.
            let delta = match &content {
                Content::Text { text, .. } if !text.is_empty() => {
                    Some(ContentDelta::TextDelta { text: text.clone() })
                }
                Content::ToolUse { input, .. } => Some(ContentDelta::InputJsonDelta {
                    partial_json: input.to_string(),
                }),
                _ => None,
            };
            self.interpret_content(
                StreamingEvent::ContentBlockStart {
                    index,
                    content_block: content,
                },
                out,
            );
            if let Some(delta) = delta {
                self.interpret_content(StreamingEvent::ContentBlockDelta { index, delta }, out);
            }
            self.interpret_content(StreamingEvent::ContentBlockStop { index }, out);
        }

        // A whole response completes the turn even without an explicit stop reason.
        let usage = PartialUsage {
            output_tokens: message.usage.output_tokens as usize,
            input_tokens: usize::try_from(message.usage.input_tokens).ok(),
            cache_creation_input_tokens: message.usage.cache_creation_input_tokens,
            cache_creation: message.usage.cache_creation,
            cache_read_input_tokens: message.usage.cache_read_input_tokens,
            output_tokens_details: message.usage.output_tokens_details,
        };
        let native = StreamingCompletionResponse {
            usage,
            stop_reason: message.stop_reason,
            stop_sequence: message.stop_sequence,
            message_id: self.message_id.clone(),
            model: self.response_model.clone(),
            provider_request_id: None,
        };
        match terminal_record(self.provider, &native) {
            Ok(record) => out.final_record(record),
            Err(error) => out.error(error),
        }
    }
}

impl Decoder<Completion> for MessagesDecoder {
    type Event = StreamingEvent;

    fn classify(&self, frame: WireFrame) -> WireEvent<StreamingEvent> {
        let data = frame.as_str();
        wire::classify_tagged_frame(&data, "type", |event_type| {
            KNOWN_EVENT_TYPES.contains(&event_type)
        })
        .map(|event| match event {
            // The one event whose payload leaves this crate as bytes rather
            // than as decoded fields, so it is captured where the frame is
            // still in hand: serde never sees the text it parsed.
            StreamingEvent::Error { error, .. } => StreamingEvent::Error {
                error,
                raw: data.to_string(),
            },
            other => other,
        })
    }

    fn interpret(&mut self, event: StreamingEvent, out: &mut AdapterOutput) {
        if self.failed {
            return;
        }

        match event {
            StreamingEvent::Message { message } => self.interpret_whole_message(message, out),
            StreamingEvent::MessageStart { message } => {
                // Bedrock-compat quirk: a `message_start` without a message
                // body is a no-op, not an error.
                let Some(message) = message else { return };
                self.input_tokens = message.usage.input_tokens;
                self.cache_creation
                    .clone_from(&message.usage.cache_creation);
                self.message_id = Some(message.id.clone());
                self.response_model = Some(message.model.clone());
            }
            StreamingEvent::MessageDelta { delta, usage } => {
                // Only a `message_delta` carrying a stop reason is the
                // provider's genuine terminal; without one it is a no-op.
                let Some(reason) = delta.stop_reason else {
                    return;
                };
                // Prefer a positive terminal input count, falling back to message_start;
                // zero-as-missing is a gateway heuristic, not a rule for cache counts.
                let usage = PartialUsage {
                    output_tokens: usage.output_tokens,
                    input_tokens: usage
                        .input_tokens
                        .filter(|tokens| *tokens > 0)
                        .or_else(|| usize::try_from(self.input_tokens).ok()),
                    cache_creation_input_tokens: usage.cache_creation_input_tokens,
                    cache_creation: usage.cache_creation.or(self.cache_creation),
                    cache_read_input_tokens: usage.cache_read_input_tokens,
                    // The terminal frame owns the output count and its breakdown.
                    output_tokens_details: usage.output_tokens_details,
                };

                let native = StreamingCompletionResponse {
                    usage,
                    stop_reason: Some(reason),
                    // Rides the same `message_delta` as the stop reason, and
                    // only that frame carries it: `message_start` always
                    // opens with `null`.
                    stop_sequence: delta.stop_sequence,
                    message_id: self.message_id.clone(),
                    model: self.response_model.clone(),
                    // Stamped by the driver onto the normalized record; the
                    // decoder never sees connection headers.
                    provider_request_id: None,
                };
                match terminal_record(self.provider, &native) {
                    Ok(record) => out.final_record(record),
                    Err(err) => out.error(err),
                }
            }
            StreamingEvent::Error { raw, .. } => {
                // Preserve the complete error envelope rather than re-encode modeled fields.
                self.failed = true;
                out.error(crate::error::ProviderError::from_provider_body(raw));
            }
            event @ (StreamingEvent::ContentBlockStart { .. }
            | StreamingEvent::ContentBlockDelta { .. }
            | StreamingEvent::ContentBlockStop { .. }
            | StreamingEvent::MessageStop
            | StreamingEvent::Ping) => self.interpret_content(event, out),
        }
    }

    fn finish(&mut self, _out: &mut AdapterOutput) {
        // EOF without `message_delta` is truncation: open blocks stay
        // partial, and no terminal record may be synthesized.
    }

    /// Messages metadata projected before normalization can discard it: the
    /// stop reason, the model, the message id, the usage and any error
    /// envelope, on the unary reply and on the stream's `message_start`,
    /// `message_delta` and `error` events.
    fn project(&self, payload: &[u8], sink: &mut dyn ObservationSink) {
        let Ok(payload) = serde_json::from_slice::<ObservedPayload>(payload) else {
            return;
        };
        let usage = payload.usage;
        let (id, model, stop_reason, nested_usage) = match payload.message {
            Some(message) => (
                message.id,
                message.model,
                message.stop_reason,
                message.usage,
            ),
            None => (payload.id, payload.model, payload.stop_reason, None),
        };
        // Anthropic reports the prompt on `message_start` and the answer's
        // running total on each `message_delta`: each is a snapshot of what it
        // knows, never a sum.
        if let Some(usage) = usage.or(nested_usage) {
            sink.emit(AdapterEvent::Usage {
                usage: AdapterUsage {
                    input_tokens: usage.input_tokens,
                    output_tokens: usage.output_tokens,
                    total_tokens: None,
                    cached_input_tokens: usage.cache_read_input_tokens,
                    reasoning_tokens: usage
                        .output_tokens_details
                        .map(|details| details.thinking_tokens),
                    tool_input_tokens: None,
                },
            });
        }
        let stop_reason = stop_reason.or(payload.delta.and_then(|delta| delta.stop_reason));
        let verdict = AdapterVerdict {
            finish_reason: stop_reason.map(|v| sink.scrub(&v)),
            block_reason: None,
            detail: None,
            model: model.map(|v| sink.scrub(&v)),
        };
        let response_id = id.map(|v| sink.scrub(&v));
        sink.provider(verdict, response_id);
        if let Some(error) = payload.error {
            error.emit(sink);
        }
    }

    fn is_finished(&self) -> bool {
        // Stop after errors so later frames cannot report a failed turn as completed.
        self.failed
    }
}

/// Observation fields from unary replies and stream events, including nested messages.
#[derive(Deserialize)]
struct ObservedPayload {
    id: Option<String>,
    model: Option<String>,
    stop_reason: Option<String>,
    usage: Option<ObservedUsage>,
    message: Option<Box<ObservedPayload>>,
    delta: Option<ObservedDelta>,
    error: Option<ObservedError>,
}

#[derive(Deserialize)]
struct ObservedUsage {
    #[serde(default, deserialize_with = "crate::observe::lenient_count")]
    input_tokens: Option<u64>,
    #[serde(default, deserialize_with = "crate::observe::lenient_count")]
    output_tokens: Option<u64>,
    #[serde(default, deserialize_with = "crate::observe::lenient_count")]
    cache_read_input_tokens: Option<u64>,
    #[serde(default)]
    output_tokens_details: Option<ObservedOutputDetails>,
}

#[derive(Deserialize)]
struct ObservedOutputDetails {
    #[serde(default)]
    thinking_tokens: u64,
}

#[derive(Deserialize)]
struct ObservedDelta {
    stop_reason: Option<String>,
}

/// Anthropic's own terminal stream record.
///
/// The adapter maps it once into the normalized [`StreamFinal`] (see
/// `terminal_record`) and serializes it onto [`StreamFinal::raw`]; callers
/// who want the provider-native shape deserialize it from there.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct StreamingCompletionResponse {
    /// Token usage carried by the terminal `message_delta` event.
    pub usage: PartialUsage,
    /// Anthropic's `stop_reason`, verbatim, when the stream reported one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
    /// Matched stop sequence reported by the terminal frame, preserved verbatim.
    /// The provider strips this sequence from output text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
    /// The `message_start` message ID, when the stream reported one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message_id: Option<String>,
    /// The model named by `message_start`, when the stream reported one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Transport request id supplied by an external record builder.
    /// Live decoders leave this absent; the driver attaches response headers
    /// to [`StreamFinal::provider_request_id`] instead.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_request_id: Option<String>,
}

/// Normalize terminal metadata for the selected `provider`, preserving the native
/// record on [`StreamFinal::raw`]. Return an error if serialization fails.
fn terminal_record(
    provider: &str,
    response: &StreamingCompletionResponse,
) -> Result<StreamFinal, ProviderError> {
    Ok(StreamFinal::new(
        provider,
        crate::completion::Usage::from(&response.usage),
        serde_json::to_value(response)?,
    )
    .with_optional_finish_reason(response.stop_reason.as_deref().map(map_finish_reason))
    .with_optional_message_id(response.message_id.clone())
    .with_optional_provider_request_id(response.provider_request_id.clone())
    .with_optional_model(response.model.clone()))
}

#[cfg(test)]
mod tests;
