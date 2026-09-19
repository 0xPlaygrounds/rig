use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::completion::{Content, Usage, anthropic_usage_totals, map_finish_reason};
use crate::completion::CompletionError;
use crate::message::ReasoningContent;
use crate::operation::{AdapterOutput, Completion};
use crate::providers::internal::wire::{self, WireEvent};
use crate::streaming::{self, BlockId, MintKind, StreamFinal, ToolCallEnd, UnparseableToolInput};
use crate::wire::{
    AdapterErrorEnvelope, AdapterEvent, AdapterUsage, AdapterVerdict, Decoder, ObservationSink,
    WireFrame,
};
use std::collections::HashMap;

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
    // The unary reply's own shape: a whole `message` object, which is what
    // `POST /v1/messages` answers without `stream`. Naming it here is the
    // ONE place the unary shape appears — it is a frame like any other, so
    // the unary and streamed paths cannot drift.
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
        /// Anthropic-compatible relays (Bedrock's Messages passthrough) can
        /// emit `message_start` with a null `message`; `None` is a no-op
        /// rather than a corrupt frame.
        #[serde(default)]
        message: Option<MessageStart>,
    },
    /// The whole message: what the endpoint answers when not streaming.
    /// Its fields are exactly `message_start`'s, plus the stop reason and
    /// usage a stream delivers on `message_delta`.
    Message {
        #[serde(flatten)]
        message: MessageStart,
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
    /// envelope.
    ///
    /// The nested `error` object is required, and that requirement is the
    /// whole of the wire's shape check: every Anthropic error body recorded
    /// under `crates/rig-cassette/fixtures/cassettes/anthropic/` nests it, and the flattened
    /// `{"type":"error","message":"…"}` form appears in no recorded traffic.
    Error {
        /// Decoding it is the whole point — it proves the body is the
        /// envelope and nothing else — but the error the consumer sees is
        /// built from `raw`, so the provider's payload rides out verbatim.
        #[allow(dead_code)]
        error: serde_json::Value,
        /// The envelope's own bytes, attached by
        /// [`MessagesDecoder::classify`] because serde cannot see them.
        ///
        /// A body rebuilt from the fields this client models is not the
        /// provider's body: re-encoding through `serde_json::Value`
        /// normalizes key order, and every sibling key of `error` is
        /// dropped — `request_id` among them, the one field a user quotes
        /// to provider support.
        #[serde(skip)]
        raw: String,
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

// Client tool-call fragment assembly lives in the shared accumulator
// (`BlockAccumulator`, fed by `ToolArguments` deltas); the adapter tracks only
// the open block's wire id. Server tool use keeps local state because its
// assembled payload becomes text-block metadata (`ANTHROPIC_RAW_CONTENT_KEY`),
// not a tool call.
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
    /// accumulator via `Reasoning` deltas; no restatement buffer exists.
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

/// The Anthropic Messages wire's [`Decoder`], for both modes.
///
/// Holds the per-reply assembly state (open tool call, server tool uses,
/// open thinking block, terminal metadata); frame-triage policy lives in
/// [`crate::driver`], not here. Every interpretation — content blocks and
/// the message-level frames alike, buffered reply included — goes through
/// [`Decoder::interpret`]: one path.
pub struct MessagesDecoder {
    /// Stable descriptor name stamped on the terminal record. An *input*
    /// rather than a constant: the Anthropic Messages stream format is
    /// shared by every Anthropic-compatible provider, so baking in
    /// `"anthropic"` here would mislabel all of them.
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
    /// A provider `error` event ended the turn; later frames are dead — the
    /// provider aborted, and interpreting more output (or a terminal) would
    /// dress the failure up as a completed turn.
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

                    // Wire quirk: the signature is not emitted as its own
                    // event — it closes the thinking block, riding on the
                    // `BlockEnd` the `content_block_stop` emits.
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
                    // Structural metadata only: a novel delta type can carry
                    // model output, which must not leak into production WARN
                    // logs (same policy as the adapter's unknown-event warn).
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
                // Keep this destructuring exhaustive so new wire fields force
                // an explicit capture-or-drop decision: block-start `text`
                // arrives via the deltas, and `cache_control` is a
                // request-side directive — both deliberately dropped here.
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
                    // `content_block_start` opens the block with its initial
                    // payload; the old `..` discarded both fields. Adaptive
                    // thinking opens with an empty `thinking`, emits no
                    // `thinking_delta` at all, and delivers the whole
                    // signature by `signature_delta` — so the block's only
                    // content is a signature, which `content_block_stop`
                    // must still close with.
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
                // Drop only a wholly empty block. A signature-only thinking
                // block (empty text, complete signature) is the
                // adaptive-thinking wire shape, and its signature is
                // replay-required provider state that Anthropic accepts back
                // verbatim (the paired non-streaming cassette replays that
                // exact empty-text signed block). The non-streaming path has
                // never gated on text, so gating here was a unary/streaming
                // divergence that silently dropped the signature.
                if let Some(thinking_state) = self.current_thinking.take() {
                    // `content_block_stop` is the wire's own lifecycle end:
                    // the shared accumulator holds the block's accumulated
                    // text, and the end carries the assembled signature
                    // (present for signed and adaptive signature-only blocks
                    // alike — replay-required provider state either way). A
                    // wholly empty block (no deltas, no signature) closes
                    // silently.
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
                                out.error(CompletionError::from(e));
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
            // Interpreted by `interpret` itself (`message_start` /
            // `message_delta` / the `error` envelope) or Known no-ops
            // (`message_stop`, `ping`).
            StreamingEvent::Message { .. }
            | StreamingEvent::MessageStart { .. }
            | StreamingEvent::MessageDelta { .. }
            | StreamingEvent::MessageStop
            | StreamingEvent::Ping
            | StreamingEvent::Error { .. } => {}
        }
    }

    /// Interpret the unary reply by *synthesizing the stream* it would have
    /// been: the same `content_block_start` / `_delta` / `_stop` frames the
    /// streaming wire sends for each content part, then the terminal the
    /// `message_delta` carries.
    ///
    /// This is why there is no second `Content -> AssistantContent` mapping
    /// and no `normalize`: the block code that assembles a streamed turn is
    /// the only code that assembles a buffered one, so the two cannot
    /// disagree about text, citations, tool arguments, thinking signatures
    /// or server tool use.
    fn interpret_whole_message(&mut self, message: MessageStart, out: &mut AdapterOutput) {
        self.input_tokens = message.usage.input_tokens;
        self.cache_creation
            .clone_from(&message.usage.cache_creation);
        self.message_id = Some(message.id);
        self.response_model = Some(message.model);

        // Anthropic has two ways to end a turn that genuinely carried no
        // content, and an empty list says exactly that:
        //
        // - `end_turn` after a tool-result round trip — documented, and it
        //   used to be normalized into a fabricated empty-text part.
        // - `stop_sequence` when the matched sequence is the first thing the
        //   model emits. Anthropic strips the sequence it stopped on, so a
        //   turn that produced nothing before it arrives with `content: []`
        //   and a 200. Rejecting that turned a completed provider turn into
        //   `EMPTY_RESPONSE_ERROR`.
        //
        // The `stop_sequence` arm additionally requires the sequence itself.
        // Every recorded stop-sequence turn names the sequence that fired, so
        // that is the full extent of the evidence; a turn claiming to have
        // stopped on a sequence while naming none is the malformed shape this
        // guard exists for, not a legal empty turn. This matters most for the
        // Anthropic-compatible gateways sharing this decoder, which are the
        // likeliest to report a stop reason without its companion field.
        //
        // Any *other* empty reply is the shared provider defect.
        //
        // The guard is deliberately asymmetric: it runs here, on the
        // buffered reply, and has no equivalent on the streamed path, which
        // still finishes such a turn cleanly with an empty choice and no
        // error. The parity this carve-out protects is for *legal* turns,
        // and widening the rejection to the stream would trade a real guard
        // for a cosmetic match — so do not "unify" it by moving it into the
        // terminal both modes share.
        let legal_empty_turn = match message.stop_reason.as_deref() {
            Some("end_turn") => true,
            Some("stop_sequence") => message.stop_sequence.is_some(),
            _ => false,
        };
        if message.content.is_empty() && !legal_empty_turn {
            self.failed = true;
            out.error(CompletionError::ResponseError(
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

        // A buffered reply is the whole turn, so its terminal is
        // unconditional — unlike a `message_delta`, which is only terminal
        // when it carries a stop reason.
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

/// The Messages wire decodes its unary and streamed replies with the same
/// state machine: `POST /v1/messages` answers with a whole `message`
/// object, which is a frame like any other, so the two modes cannot drift.
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
                // cache_creation_input_tokens and cache_read_input_tokens are
                // cumulative totals on message_delta.usage per the Anthropic
                // streaming API spec — use them directly.
                //
                // `input_tokens` prefers the terminal `message_delta` and falls
                // back to `message_start`.
                //
                // Anthropic proper sends the count on *both* frames and they
                // agree (every recorded cassette under
                // `crates/rig-cassette/fixtures/cassettes/anthropic/` reporting it on the delta reports
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
                // The provider aborted the turn in-band. The envelope is the
                // error body verbatim — every field it carried, in the order
                // it carried them — so the consumer reads what Anthropic
                // said rather than what this client models. The stream
                // carries it as an in-band `Err` item, and EOF without
                // `message_delta` then withholds the terminal record.
                self.failed = true;
                out.error(crate::provider_response::completion_error_from_body(raw));
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
                    reasoning_tokens: None,
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
            sink.emit(AdapterEvent::ErrorEnvelope {
                error: AdapterErrorEnvelope {
                    code: None,
                    status: error.kind.map(|v| sink.scrub(&v)),
                    message: error.message.map(|v| sink.scrub(&v)),
                },
            });
        }
    }

    fn is_finished(&self) -> bool {
        // A provider `error` event is the wire's own terminal failure:
        // `interpret` already pushed the in-band `Err`, so the driver must
        // stop reading — a later modeled frame (e.g. a stray `message_delta`)
        // would otherwise dress the aborted turn up as a completed one.
        self.failed
    }
}

/// One object covers every payload `project` above sees — the unary reply and
/// each stream event: a `message_start` nests the message, a `message_delta`
/// carries the stop reason under `delta` and the cumulative output usage
/// beside it.
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
}

#[derive(Deserialize)]
struct ObservedDelta {
    stop_reason: Option<String>,
}

#[derive(Deserialize)]
struct ObservedError {
    #[serde(rename = "type")]
    kind: Option<String>,
    message: Option<String>,
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
    /// response header — not part of any stream frame. The adapter never
    /// sees connection headers, so on a live stream this is `None` and the
    /// transport stamps the id onto the normalized
    /// [`StreamFinal::provider_request_id`] instead; the field survives for
    /// records built elsewhere (and re-normalizes through
    /// `terminal_record`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_request_id: Option<String>,
}

/// Normalize an Anthropic terminal stream record for `provider`, keeping
/// the native record on [`StreamFinal::raw`].
///
/// The provider descriptor name is an *input* rather than a constant: the
/// Anthropic Messages stream format is shared by every Anthropic-compatible
/// provider, so baking in `"anthropic"` here would mislabel all of them.
fn terminal_record(
    provider: &str,
    response: &StreamingCompletionResponse,
) -> Result<StreamFinal, CompletionError> {
    Ok(
        StreamFinal::new(provider, crate::completion::Usage::from(&response.usage))
            .with_optional_finish_reason(response.stop_reason.as_deref().map(map_finish_reason))
            .with_optional_message_id(response.message_id.clone())
            .with_optional_provider_request_id(response.provider_request_id.clone())
            .with_optional_model(response.model.clone())
            .with_raw(serde_json::to_value(response)?),
    )
}

#[cfg(test)]
mod tests;
