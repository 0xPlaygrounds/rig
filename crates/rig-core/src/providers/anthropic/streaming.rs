use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::completion::{
    CompletionResponse, Content, TextExtras, anthropic_usage_totals, map_finish_reason,
};
use crate::error::ProviderError;
use crate::message::ReasoningContent;
use crate::observe::ObservedError;
use crate::operation::{
    CallFragment, Completion, Finish, IfMalformed, ReasoningPart, Seal, TextPart,
};
use crate::providers::internal::wire;
use crate::wire::{
    AdapterEvent, AdapterUsage, AdapterVerdict, Decoder, Flow, ObservationSink, Out, WireEvent,
    WireFrame,
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

/// A block this crate does not model, assembled from its start frame and
/// its deltas. Deltas fold by one rule for every such block: `partial_json`
/// fragments replace `input` when the block stops; any other delta appends
/// its string fields to the block's same-named fields and sets the rest.
struct ProviderBlock {
    block: serde_json::Map<String, Value>,
    input_json: String,
}

impl ProviderBlock {
    fn absorb(&mut self, delta: ContentDelta) -> Result<(), ProviderError> {
        match delta {
            ContentDelta::InputJsonDelta { partial_json } => {
                self.input_json.push_str(&partial_json);
            }
            ContentDelta::TextDelta { text } => self.append("text", text),
            ContentDelta::ThinkingDelta { thinking } => self.append("thinking", thinking),
            ContentDelta::SignatureDelta { signature } => self.append("signature", signature),
            ContentDelta::CitationsDelta { citation } => {
                let citation = serde_json::to_value(citation)?;
                match self.block.get_mut("citations") {
                    Some(Value::Array(citations)) => citations.push(citation),
                    _ => {
                        self.block
                            .insert("citations".to_owned(), Value::Array(vec![citation]));
                    }
                }
            }
            ContentDelta::Unknown(Value::Object(fields)) => {
                for (key, value) in fields {
                    match value {
                        _ if key == "type" => {}
                        Value::String(fragment) => self.append(&key, fragment),
                        value => {
                            self.block.insert(key, value);
                        }
                    }
                }
            }
            ContentDelta::Unknown(_) => {}
        }
        Ok(())
    }

    fn append(&mut self, key: &str, fragment: String) {
        match self.block.get_mut(key) {
            Some(Value::String(existing)) => existing.push_str(&fragment),
            _ => {
                self.block.insert(key.to_owned(), Value::String(fragment));
            }
        }
    }

    /// The block as the API would state it whole. Accumulated input that
    /// does not parse fails the reply: the stop promised a complete block.
    fn finish(mut self) -> Result<crate::message::Verbatim, ProviderError> {
        if !self.input_json.is_empty() {
            let input = serde_json::from_str(&self.input_json)?;
            self.block.insert("input".to_owned(), input);
        }
        Ok(crate::message::Verbatim::new(self.block))
    }
}

/// Decode Messages replies, a whole message or a stream of events.
/// EOF without a `message_delta` stop reason is truncation.
pub struct MessagesDecoder<'id> {
    /// The text part of each open content block.
    texts: HashMap<usize, TextPart<'id>>,
    /// The thinking part of each open content block, and its signature:
    /// fragments win over the opening value, and an absent one is `None`.
    thinking: HashMap<usize, (ReasoningPart<'id>, String, String)>,
    /// The content block of the open client tool call, whose index its
    /// fragments are buffered under.
    current_tool_call: Option<usize>,
    /// The open blocks this crate does not model, by content-block index.
    provider_blocks: HashMap<usize, ProviderBlock>,
    input_tokens: u64,
    /// Per-TTL cache-write breakdown from `message_start`; the terminal
    /// `message_delta` usage omits it.
    cache_creation: Option<super::completion::CacheCreation>,
    /// Cache reads and writes from `message_start`, for a terminal
    /// `message_delta` that does not repeat them: rig's input counts them.
    cache_read_input_tokens: Option<u64>,
    cache_creation_input_tokens: Option<u64>,
    message_id: Option<String>,
    response_model: Option<String>,
}

impl Default for MessagesDecoder<'_> {
    fn default() -> Self {
        Self::new()
    }
}

impl MessagesDecoder<'_> {
    /// A fresh decoder for one reply.
    pub fn new() -> Self {
        Self {
            texts: HashMap::new(),
            thinking: HashMap::new(),
            current_tool_call: None,
            provider_blocks: HashMap::new(),
            input_tokens: 0,
            cache_creation: None,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
            message_id: None,
            response_model: None,
        }
    }
}

impl<'id> MessagesDecoder<'id> {
    /// The content-block frames: `content_block_start` / `_delta` / `_stop`.
    fn interpret_content(
        &mut self,
        event: StreamingEvent,
        out: &mut Out<'id, Completion>,
    ) -> Result<(), ProviderError> {
        match event {
            StreamingEvent::ContentBlockDelta { index, delta } => match delta {
                delta if self.provider_blocks.contains_key(&index) => {
                    if let Some(block) = self.provider_blocks.get_mut(&index) {
                        block.absorb(delta)?;
                    }
                }
                ContentDelta::TextDelta { text } => {
                    if self.current_tool_call.is_none() {
                        let part = self.texts.entry(index).or_insert_with(|| out.text());
                        out.push_text(part, &text);
                    }
                }
                ContentDelta::InputJsonDelta { partial_json } => {
                    if let Some(call) = self.current_tool_call {
                        out.call_fragment(
                            call,
                            CallFragment {
                                arguments: Some(partial_json.as_str()),
                                ..CallFragment::default()
                            },
                        )?;
                    }
                }
                ContentDelta::ThinkingDelta { thinking } => {
                    let (part, _, _) = self
                        .thinking
                        .entry(index)
                        .or_insert_with(|| (out.reasoning(), String::new(), String::new()));
                    out.push_reasoning(part, &thinking);
                }
                ContentDelta::SignatureDelta { signature } => {
                    let (_, fragments, _) = self
                        .thinking
                        .entry(index)
                        .or_insert_with(|| (out.reasoning(), String::new(), String::new()));
                    // The completed signature closes the thinking part.
                    fragments.push_str(&signature);
                }
                ContentDelta::CitationsDelta { citation } => {
                    let part = self.texts.entry(index).or_insert_with(|| out.text());
                    out.text_extras(
                        part,
                        TextExtras {
                            citations: vec![citation],
                        }
                        .into(),
                    );
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
            } => {
                // Blocks stream one after another, so a block that starts
                // completes every earlier one still open.
                self.flush_provider_blocks(Some(index), out)?;
                self.start_block(index, content_block, out)?;
            }
            StreamingEvent::ContentBlockStop { index } => {
                // Signature-only thinking parts carry provider state required
                // for replay.
                if let Some((part, fragments, initial)) = self.thinking.remove(&index) {
                    let signature = if fragments.is_empty() {
                        initial
                    } else {
                        fragments
                    };
                    out.close_reasoning(
                        part,
                        Seal {
                            signature: (!signature.is_empty()).then_some(signature),
                            ..Seal::default()
                        },
                    );
                    return Ok(());
                }

                if let Some(block) = self.provider_blocks.remove(&index) {
                    out.provider_item(crate::message::ProviderItem::AnthropicMessages(
                        block.finish()?,
                    ));
                    return Ok(());
                }

                // `content_block_stop` promises a complete block: empty input
                // finalizes to `{}`, and malformed input fails the reply.
                if self.current_tool_call == Some(index) {
                    self.current_tool_call = None;
                    out.close_pending(index, IfMalformed::Fail)?;
                    return Ok(());
                }

                if let Some(part) = self.texts.remove(&index) {
                    out.close_text(part);
                }
            }
            StreamingEvent::Message { .. }
            | StreamingEvent::MessageStart { .. }
            | StreamingEvent::MessageDelta { .. }
            | StreamingEvent::MessageStop
            | StreamingEvent::Ping
            | StreamingEvent::Error { .. } => {}
        }
        Ok(())
    }

    /// Open the part a `content_block_start` names.
    fn start_block(
        &mut self,
        index: usize,
        content_block: Content,
        out: &mut Out<'id, Completion>,
    ) -> Result<(), ProviderError> {
        match content_block {
            // Text arrives through deltas; cache_control is request-only metadata.
            Content::Text {
                text: _,
                citations,
                cache_control: _,
            } => {
                let part = out.text();
                if !citations.is_empty() {
                    out.text_extras(&part, TextExtras { citations }.into());
                }
                self.texts.insert(index, part);
            }
            // Every block this crate does not model, hosted-tool calls
            // and results included, is kept whole in its place.
            Content::Unknown(block) => {
                self.provider_blocks.insert(
                    index,
                    ProviderBlock {
                        block: block.into_map(),
                        input_json: String::new(),
                    },
                );
            }
            Content::ToolUse { id, name, .. } => {
                self.current_tool_call = Some(index);
                out.call_fragment(
                    index,
                    CallFragment {
                        id: Some(id.as_str()),
                        name: Some(name.as_str()),
                        ..CallFragment::default()
                    },
                )?;
            }
            Content::Thinking {
                thinking,
                signature,
            } => {
                // Adaptive thinking may carry only a signature, so the
                // part opens even when the opening text is empty.
                let part = out.reasoning();
                out.push_reasoning(&part, &thinking);
                self.thinking
                    .insert(index, (part, String::new(), signature.unwrap_or_default()));
            }
            Content::RedactedThinking { data } => {
                out.reasoning_block(crate::message::Reasoning {
                    id: None,
                    content: vec![ReasoningContent::Redacted { data }],
                });
            }
            // Request-side content kinds; an assistant stream never
            // opens a block with them, and there is nothing to write.
            Content::Image { .. } | Content::ToolResult { .. } | Content::Document { .. } => {}
        }
        Ok(())
    }

    /// Write the open provider blocks below `before` (all of them for
    /// `None`) in block order.
    fn flush_provider_blocks(
        &mut self,
        before: Option<usize>,
        out: &mut Out<'id, Completion>,
    ) -> Result<(), ProviderError> {
        let mut done: Vec<usize> = self
            .provider_blocks
            .keys()
            .copied()
            .filter(|index| before.is_none_or(|before| *index < before))
            .collect();
        done.sort_unstable();
        for index in done {
            if let Some(block) = self.provider_blocks.remove(&index) {
                out.provider_item(crate::message::ProviderItem::AnthropicMessages(
                    block.finish()?,
                ));
            }
        }
        Ok(())
    }

    /// A whole message, written block by block as a stream states it, then
    /// ended. Empty content is refused unless the stop reason is `end_turn`,
    /// or `stop_sequence` with a reported sequence.
    fn interpret_whole_message(
        &mut self,
        message: CompletionResponse,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        self.input_tokens = message.usage.input_tokens;
        self.cache_creation
            .clone_from(&message.usage.cache_creation);
        self.message_id = Some(message.id);
        self.response_model = Some(message.model);

        // An empty end_turn and a stripped stop sequence are valid answers.
        let legal_empty_turn = match message.stop_reason.as_deref() {
            Some("end_turn") => true,
            Some("stop_sequence") => message.stop_sequence.is_some(),
            _ => false,
        };
        if message.content.is_empty() && !legal_empty_turn {
            return Err(ProviderError::Response(
                crate::message::EMPTY_RESPONSE_ERROR.to_owned(),
            ));
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
                &mut out,
            )?;
            if let Some(delta) = delta {
                self.interpret_content(
                    StreamingEvent::ContentBlockDelta { index, delta },
                    &mut out,
                )?;
            }
            self.interpret_content(StreamingEvent::ContentBlockStop { index }, &mut out)?;
        }

        self.flush_provider_blocks(None, &mut out)?;
        // A whole message completes the turn even without an explicit stop
        // reason. Its `raw` is the message itself.
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
        };
        Ok(out.end(finish_of(&native)))
    }
}

impl<'id> Decoder<'id, Completion> for MessagesDecoder<'id> {
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

    fn decode(
        &mut self,
        event: StreamingEvent,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        match event {
            StreamingEvent::Message { message } => self.interpret_whole_message(message, out),
            StreamingEvent::MessageStart { message } => {
                // Bedrock-compat quirk: a `message_start` without a message
                // body is a no-op, not an error.
                if let Some(message) = message {
                    self.input_tokens = message.usage.input_tokens;
                    self.cache_creation
                        .clone_from(&message.usage.cache_creation);
                    self.cache_read_input_tokens = message.usage.cache_read_input_tokens;
                    self.cache_creation_input_tokens = message.usage.cache_creation_input_tokens;
                    self.message_id = Some(message.id.clone());
                    self.response_model = Some(message.model.clone());
                }
                Ok(Flow::More)
            }
            StreamingEvent::MessageDelta { delta, usage } => {
                // Only a `message_delta` carrying a stop reason is the
                // provider's end; without one it is a no-op.
                let Some(reason) = delta.stop_reason else {
                    return Ok(Flow::More);
                };
                self.flush_provider_blocks(None, &mut out)?;
                // Prefer a positive terminal input count, falling back to message_start;
                // zero-as-missing is a gateway heuristic, not a rule for cache counts.
                let usage = PartialUsage {
                    output_tokens: usage.output_tokens,
                    input_tokens: usage
                        .input_tokens
                        .filter(|tokens| *tokens > 0)
                        .or_else(|| usize::try_from(self.input_tokens).ok()),
                    // A terminal frame that omits the cache counters keeps
                    // `message_start`'s, so input still counts the cache.
                    cache_creation_input_tokens: usage
                        .cache_creation_input_tokens
                        .or(self.cache_creation_input_tokens),
                    cache_creation: usage.cache_creation.or(self.cache_creation),
                    cache_read_input_tokens: usage
                        .cache_read_input_tokens
                        .or(self.cache_read_input_tokens),
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
                };
                out.raw(serde_json::to_value(&native)?);
                Ok(out.end(finish_of(&native)))
            }
            StreamingEvent::Error { raw, .. } => {
                // Preserve the complete error envelope rather than re-encode modeled fields.
                Err(crate::error::ProviderError::from_provider_body(raw))
            }
            event @ (StreamingEvent::ContentBlockStart { .. }
            | StreamingEvent::ContentBlockDelta { .. }
            | StreamingEvent::ContentBlockStop { .. }
            | StreamingEvent::MessageStop
            | StreamingEvent::Ping) => {
                self.interpret_content(event, &mut out)?;
                Ok(Flow::More)
            }
        }
    }
}

impl MessagesDecoder<'_> {
    /// Messages metadata projected before normalization can discard it: the
    /// stop reason, the model, the message id, the usage and any error
    /// envelope, on the unary reply and on the stream's `message_start`,
    /// `message_delta` and `error` events.
    pub(crate) fn project(payload: &[u8], sink: &mut ObservationSink<'_>) {
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

/// Anthropic's own terminal stream record, a streamed response's `raw`:
/// callers who want the provider-native shape deserialize it from there.
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
}

/// The provider's end of the reply, from Anthropic's terminal record.
fn finish_of(response: &StreamingCompletionResponse) -> Finish {
    Finish {
        usage: crate::completion::Usage::from(&response.usage),
        reason: response.stop_reason.as_deref().map(map_finish_reason),
        message_id: response.message_id.clone(),
        model: response.model.clone(),
        ..Finish::default()
    }
}

#[cfg(test)]
mod tests;
