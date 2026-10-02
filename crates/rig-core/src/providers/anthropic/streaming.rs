use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::completion::{
    CompletionResponse, ContentItem, anthropic_usage_totals, map_finish_reason,
};
use crate::error::ProviderError;
use crate::message::{CallId, ToolName};
use crate::observe::ObservedError;
use crate::operation::{Block, Completion, Finish, IfMalformed};
use crate::providers::internal::wire;
use crate::wire::{
    AdapterEvent, AdapterUsage, AdapterVerdict, Decoder, Flow, ObservationSink, Out, WireEvent,
    WireFrame,
};
use std::collections::HashMap;

/// Recognized Messages event tags. Listed events must decode fully;
/// unlisted tags classify as unknown.
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
    /// A content block opens, as the provider states it before its deltas.
    ContentBlockStart {
        index: usize,
        content_block: ContentItem,
    },
    /// A delta to an open content block, as the provider sent it.
    ContentBlockDelta {
        index: usize,
        delta: ContentItem,
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

#[derive(Debug, Deserialize)]
pub struct MessageDelta {
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    /// What stopped the model beyond `stop_reason`, such as a refusal's
    /// explanation.
    #[serde(default)]
    pub stop_details: Option<Value>,
    /// The code-execution container the reply ran in.
    #[serde(default)]
    pub container: Option<Value>,
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

/// Decodes Messages replies, a whole message or a stream of events: each
/// content block becomes one block, in wire order, whose provider item is
/// the block as it ended. EOF without a `message_delta` stop reason is
/// truncation.
#[derive(Debug, Default)]
pub struct MessagesDecoder {
    /// The input JSON streamed so far for each open block that takes
    /// `input_json_delta`, and whether the block is a call.
    inputs: HashMap<usize, (bool, String)>,
    /// Whether a block other than a leading `fallback` marker opened.
    opened: bool,
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
    container: Option<Value>,
}

impl MessagesDecoder {
    /// A fresh decoder for one reply.
    pub fn new() -> Self {
        Self::default()
    }

    /// Open the content block at `index` as the provider states it.
    fn start(
        &mut self,
        index: usize,
        block: ContentItem,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        // A leading `fallback` names the model that took over; one after
        // output began is a fallback rig cannot represent (pi's rule).
        if block.kind() == "fallback" {
            if self.opened {
                return Err(ProviderError::Response(
                    "Anthropic performed an unsupported mid-output model fallback".to_owned(),
                ));
            }
            return out.open(
                index,
                Block::Opaque { replay: false },
                Value::Object(block.0),
            );
        }
        self.opened = true;
        let (opened, text) = match block.kind() {
            "text" => (Block::Text, block.str("text").to_owned()),
            "thinking" => (
                Block::Reasoning { redacted: false },
                block.str("thinking").to_owned(),
            ),
            "redacted_thinking" => (Block::Reasoning { redacted: true }, String::new()),
            "tool_use" => {
                let id = CallId::from_wire(block.str("id"));
                let name = ToolName::new(block.str("name")).map_err(|error| {
                    ProviderError::Response(format!("Anthropic `tool_use`: {error}"))
                })?;
                let input = block.0.get("input").cloned().unwrap_or_default();
                self.inputs.insert(index, (true, String::new()));
                out.open(index, Block::Call { id, name }, Value::Object(block.0))?;
                // A whole reply states the input on the block; a stream
                // streams it, and the fragments win.
                return out.announce(index, input);
            }
            _ => {
                self.inputs.insert(index, (false, String::new()));
                (Block::Opaque { replay: true }, String::new())
            }
        };
        out.open(index, opened, Value::Object(block.0))?;
        out.push(index, &text)
    }

    /// Apply a delta to the open block at `index`: text and reasoning grow
    /// both the block and its item, input JSON is assembled, a citation is
    /// appended, and any other delta merges into the item by key.
    fn delta(
        &mut self,
        index: usize,
        delta: ContentItem,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        match delta.kind() {
            kind @ ("text_delta" | "thinking_delta") => {
                let key = if kind == "text_delta" {
                    "text"
                } else {
                    "thinking"
                };
                out.push(index, delta.str(key))?;
                out.merge(index, &delta.0)
            }
            "input_json_delta" => {
                let fragment = delta.str("partial_json");
                let Some((call, json)) = self.inputs.get_mut(&index) else {
                    return Err(ProviderError::Response(format!(
                        "Anthropic streamed input to content block {index}, which takes none"
                    )));
                };
                json.push_str(fragment);
                if *call {
                    out.push(index, fragment)
                } else {
                    Ok(())
                }
            }
            "citations_delta" => {
                let citation = delta.0.get("citation").cloned().unwrap_or_default();
                out.edit(index, |item| {
                    if let Some(item) = item.as_object_mut() {
                        match item.get_mut("citations") {
                            Some(Value::Array(citations)) => citations.push(citation),
                            _ => {
                                item.insert("citations".to_owned(), Value::Array(vec![citation]));
                            }
                        }
                    }
                })
            }
            // Signatures concatenate; `compaction_delta` and kinds rig has
            // never seen land in the item too.
            _ => out.merge(index, &delta.0),
        }
    }

    /// Close the block at `index`, its streamed input set on its item.
    fn stop(&mut self, index: usize, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
        if let Some((call, json)) = self.inputs.remove(&index)
            && !json.is_empty()
        {
            match crate::json_utils::parse_tool_arguments(&json) {
                Ok(input) => out.edit(index, |item| {
                    if let Some(item) = item.as_object_mut() {
                        item.insert("input".to_owned(), input);
                    }
                })?,
                // The close reports a call's malformed input with its id.
                Err(_) if call => {}
                Err(error) => {
                    return Err(ProviderError::Response(format!(
                        "Anthropic content block {index} streamed malformed input: {error}"
                    )));
                }
            }
        }
        // `content_block_stop` promises a complete block: malformed input
        // fails the reply.
        out.close(index, IfMalformed::Fail)
    }

    fn metadata(&mut self, message: &CompletionResponse) {
        self.input_tokens = message.usage.input_tokens;
        self.cache_creation
            .clone_from(&message.usage.cache_creation);
        self.cache_read_input_tokens = message.usage.cache_read_input_tokens;
        self.cache_creation_input_tokens = message.usage.cache_creation_input_tokens;
        self.message_id = Some(message.id.clone());
        self.response_model = Some(message.model.clone());
        self.note_container(message.container.as_ref());
    }

    fn note_container(&mut self, container: Option<&Value>) {
        if let Some(container) = container.filter(|container| !container.is_null()) {
            self.container = Some(container.clone());
        }
    }

    /// End the reply with Anthropic's terminal record.
    fn end(
        &self,
        native: &StreamingCompletionResponse,
        details: Option<&Value>,
        mut out: Out<'_, Completion>,
    ) -> Flow {
        if let Some(container) = &self.container {
            out.message_native(serde_json::json!({ "container": container }));
        }
        // A refusal fails the turn with its explanation (pi's rule).
        let error = match native.stop_reason.as_deref() {
            Some("refusal") => Some(
                details
                    .and_then(|details| details.get("explanation"))
                    .and_then(Value::as_str)
                    .filter(|explanation| !explanation.is_empty())
                    .unwrap_or("The model refused to complete the request")
                    .to_owned(),
            ),
            Some("sensitive") => Some("Provider stopped with: sensitive".to_owned()),
            _ => None,
        };
        out.end(Finish {
            usage: crate::completion::Usage::from(&native.usage),
            reason: native.stop_reason.as_deref().map(map_finish_reason),
            response_id: native.message_id.clone(),
            model: native.model.clone(),
            error,
        })
    }

    /// A whole message, written block by block through the calls a stream
    /// makes, then ended. Empty content is refused unless the stop reason
    /// is `end_turn`, or `stop_sequence` with a reported sequence.
    fn whole(
        &mut self,
        message: CompletionResponse,
        mut out: Out<'_, Completion>,
    ) -> Result<Flow, ProviderError> {
        self.metadata(&message);
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
        for (index, block) in message.content.into_iter().enumerate() {
            self.start(index, block, &mut out)?;
            self.stop(index, &mut out)?;
        }
        let native = StreamingCompletionResponse {
            usage: PartialUsage {
                output_tokens: message.usage.output_tokens as usize,
                input_tokens: usize::try_from(message.usage.input_tokens).ok(),
                cache_creation_input_tokens: message.usage.cache_creation_input_tokens,
                cache_creation: message.usage.cache_creation,
                cache_read_input_tokens: message.usage.cache_read_input_tokens,
                output_tokens_details: message.usage.output_tokens_details,
            },
            stop_reason: message.stop_reason,
            stop_sequence: message.stop_sequence,
            message_id: self.message_id.clone(),
            model: self.response_model.clone(),
        };
        Ok(self.end(&native, message.stop_details.as_ref(), out))
    }
}

impl<'id> Decoder<'id, Completion> for MessagesDecoder {
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

    #[deny(clippy::wildcard_enum_match_arm)]
    fn decode(
        &mut self,
        event: StreamingEvent,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        match event {
            StreamingEvent::Message { message } => return self.whole(message, out),
            StreamingEvent::MessageStart { message } => {
                // Bedrock-compat quirk: a `message_start` without a message
                // body is a no-op, not an error.
                if let Some(message) = message {
                    self.metadata(&message);
                }
            }
            StreamingEvent::ContentBlockStart {
                index,
                content_block,
            } => self.start(index, content_block, &mut out)?,
            StreamingEvent::ContentBlockDelta { index, delta } => {
                self.delta(index, delta, &mut out)?;
            }
            StreamingEvent::ContentBlockStop { index } => self.stop(index, &mut out)?,
            StreamingEvent::MessageDelta { delta, usage } => {
                self.note_container(delta.container.as_ref());
                // Only a `message_delta` carrying a stop reason is the
                // provider's end; without one it is a no-op.
                let Some(reason) = delta.stop_reason else {
                    return Ok(Flow::More);
                };
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
                return Ok(self.end(&native, delta.stop_details.as_ref(), out));
            }
            StreamingEvent::Error { raw, .. } => {
                // Preserve the complete error envelope rather than re-encode modeled fields.
                return Err(crate::error::ProviderError::from_provider_body(raw));
            }
            StreamingEvent::MessageStop | StreamingEvent::Ping => {}
        }
        Ok(Flow::More)
    }
}

impl MessagesDecoder {
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

#[cfg(test)]
mod tests;
