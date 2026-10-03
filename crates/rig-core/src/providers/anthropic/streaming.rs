use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::completion::{CacheCreation, OutputTokensDetails};
use crate::completion::FinishReason;
use crate::error::ProviderError;
use crate::message::{CallId, ToolName};
use crate::observe::ObservedError;
use crate::operation::{Block, CallFragment, Completion, Finish};
use crate::providers::internal::wire;
use crate::wire::{
    AdapterEvent, AdapterUsage, AdapterVerdict, Decoder, Flow, ObservationSink, Out, WireEvent,
    WireFrame,
};
use std::collections::HashMap;

/// Recognized Messages event tags; any other tag classifies as unknown.
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

/// One Messages event, or the whole message a unary reply is, as the
/// provider sent it. The decoder reads each field it needs on its own, so
/// an invented field, or a known one of another type, never fails a reply.
#[derive(Debug, Clone, PartialEq)]
pub struct MessagesEvent {
    /// The event's fields.
    pub fields: Map<String, Value>,
    /// The frame's text when it is an `error` event, whose envelope is
    /// reported verbatim.
    raw: Option<String>,
}

impl MessagesEvent {
    fn kind(&self) -> &str {
        str_of(&self.fields, "type")
    }

    /// The content block the event addresses. A block event without one
    /// cannot be applied.
    fn index(&self) -> Result<usize, ProviderError> {
        self.fields
            .get("index")
            .and_then(Value::as_u64)
            .and_then(|index| usize::try_from(index).ok())
            .ok_or_else(|| {
                ProviderError::Response(format!("Anthropic `{}` names no block index", self.kind()))
            })
    }

    /// The object under `key`, the block or delta the event carries. One
    /// without a `type` cannot be told apart from any other.
    fn item(&self, key: &str) -> Result<Map<String, Value>, ProviderError> {
        match self.fields.get(key) {
            Some(Value::Object(item)) if item.get("type").is_some_and(Value::is_string) => {
                Ok(item.clone())
            }
            _ => Err(ProviderError::Response(format!(
                "Anthropic `{}` carries no `{key}` with a string `type`",
                self.kind()
            ))),
        }
    }
}

/// The string field `key`, empty when absent or of another type.
fn str_of<'a>(fields: &'a Map<String, Value>, key: &str) -> &'a str {
    fields.get(key).and_then(Value::as_str).unwrap_or_default()
}

/// The string field `key`, when present.
fn string_of(fields: &Map<String, Value>, key: &str) -> Option<String> {
    fields.get(key).and_then(Value::as_str).map(str::to_owned)
}

/// Anthropic's usage counters, read leniently: a counter that is absent or
/// not a count is unknown. The decoder and the observation projection both
/// read usage through [`Counts::of`].
#[derive(Debug, Clone, Default, PartialEq)]
struct Counts {
    input: Option<u64>,
    output: Option<u64>,
    cache_read: Option<u64>,
    cache_creation: Option<u64>,
    /// The per-TTL breakdown of `cache_creation`, as sent.
    cache_creation_split: Option<Value>,
    thinking: Option<u64>,
}

impl Counts {
    fn of(usage: Option<&Value>) -> Self {
        let count = |pointer: &str| usage.and_then(|usage| usage.pointer(pointer)?.as_u64());
        Self {
            input: count("/input_tokens"),
            output: count("/output_tokens"),
            cache_read: count("/cache_read_input_tokens"),
            cache_creation: count("/cache_creation_input_tokens"),
            cache_creation_split: usage
                .and_then(|usage| usage.get("cache_creation"))
                .filter(|split| split.is_object())
                .cloned(),
            thinking: count("/output_tokens_details/thinking_tokens"),
        }
    }

    /// Rig's usage: its input is `input_tokens` plus the cache reads and
    /// writes counted beside it, its output `output_tokens` (thinking
    /// included), and its total their sum when both are known.
    fn usage(&self) -> crate::completion::Usage {
        let input = self.input.map(|uncached| {
            uncached + self.cache_read.unwrap_or(0) + self.cache_creation.unwrap_or(0)
        });
        crate::completion::Usage {
            input_tokens: input,
            output_tokens: self.output,
            cached_input_tokens: self.cache_read,
            cache_creation_input_tokens: self.cache_creation,
            reasoning_tokens: self.thinking,
            total_tokens: input.zip(self.output).map(|(input, output)| input + output),
            tool_use_prompt_tokens: None,
        }
    }

    /// The counters as [`PartialUsage`] spells them in a stream's `raw`.
    fn partial(&self) -> PartialUsage {
        PartialUsage {
            output_tokens: self
                .output
                .and_then(|tokens| usize::try_from(tokens).ok())
                .unwrap_or_default(),
            input_tokens: self.input.and_then(|tokens| usize::try_from(tokens).ok()),
            cache_creation_input_tokens: self.cache_creation,
            cache_creation: self
                .cache_creation_split
                .as_ref()
                .and_then(|split| CacheCreation::deserialize(split).ok()),
            cache_read_input_tokens: self.cache_read,
            output_tokens_details: self
                .thinking
                .map(|thinking_tokens| OutputTokensDetails { thinking_tokens }),
        }
    }
}

/// How a Messages `stop_reason` ends the turn, and the error a failed one
/// reports. Every reason Anthropic documents is listed; any other value is
/// [`FinishReason::Other`], which fails the turn.
fn finish_of(reason: &str, details: Option<&Value>) -> (FinishReason, Option<String>) {
    match reason {
        // `pause_turn` is a server-tool loop that stopped at its limit: the
        // turn is replayed as it is to resume it (pi's rule).
        "end_turn" | "stop_sequence" | "pause_turn" => (FinishReason::Stop, None),
        // The output reached `max_tokens`, or filled the context window:
        // either way the turn holds what was produced.
        "max_tokens" | "model_context_window_exceeded" => (FinishReason::Length, None),
        "tool_use" => (FinishReason::ToolCalls, None),
        // A refusal fails the turn with its explanation (pi's rule).
        "refusal" => (
            FinishReason::ContentFilter,
            Some(
                details
                    .and_then(|details| details.get("explanation"))
                    .and_then(Value::as_str)
                    .filter(|explanation| !explanation.is_empty())
                    .unwrap_or("The model refused to complete the request")
                    .to_owned(),
            ),
        ),
        // A gateway's safety stop, which pi also handles.
        "sensitive" => (
            FinishReason::Other(reason.to_owned()),
            Some("Provider stopped with: sensitive".to_owned()),
        ),
        other => (FinishReason::Other(other.to_owned()), None),
    }
}

/// The counters a stream's terminal `message_delta` reports, a typed view
/// of a streamed response's `raw`. The decoder never reads it.
#[derive(Debug, Deserialize, Clone, Serialize, Default)]
pub struct PartialUsage {
    pub output_tokens: usize,
    #[serde(default)]
    pub input_tokens: Option<usize>,
    #[serde(default)]
    pub cache_creation_input_tokens: Option<u64>,
    /// Per-TTL breakdown of `cache_creation_input_tokens`. Anthropic reports
    /// it on `message_start`, not the terminal `message_delta`; the decoder
    /// carries it forward onto the terminal usage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation: Option<CacheCreation>,
    #[serde(default)]
    pub cache_read_input_tokens: Option<u64>,
    /// Output-token breakdown reported by the terminal `message_delta`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<OutputTokensDetails>,
}

/// Decodes Messages replies, a whole message or a stream of events: each
/// content block becomes one block, in wire order, whose provider item is
/// the block as `content_block_stop` states it. EOF without a
/// `message_delta` stop reason is truncation.
#[derive(Debug, Default)]
pub struct MessagesDecoder {
    /// The input JSON streamed so far for each open block, and whether the
    /// block is a call.
    inputs: HashMap<usize, (bool, String)>,
    /// Whether a block other than a leading `fallback` marker opened.
    opened: bool,
    /// The counters `message_start` reported, for a terminal `message_delta`
    /// that does not repeat them.
    start: Counts,
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
        block: Map<String, Value>,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        let kind = str_of(&block, "type").to_owned();
        // A leading `fallback` names the model that took over; one after
        // output began is a fallback rig cannot represent (pi's rule).
        if kind == "fallback" {
            if self.opened {
                return Err(ProviderError::Response(
                    "Anthropic performed an unsupported mid-output model fallback".to_owned(),
                ));
            }
            return out.open(index, Block::Opaque { replay: false }, Value::Object(block));
        }
        self.opened = true;
        let (opened, text) = match kind.as_str() {
            "text" => (Block::Text, str_of(&block, "text").to_owned()),
            "thinking" => (
                Block::Reasoning { redacted: false },
                str_of(&block, "thinking").to_owned(),
            ),
            "redacted_thinking" => (Block::Reasoning { redacted: true }, String::new()),
            "tool_use" => {
                self.inputs.insert(index, (true, String::new()));
                let id = str_of(&block, "id");
                let input = block.get("input").cloned().unwrap_or_default();
                match ToolName::new(str_of(&block, "name")) {
                    Ok(name) => out.open(
                        index,
                        Block::Call {
                            id: CallId::from_wire(id),
                            name,
                        },
                        Value::Object(block),
                    )?,
                    // A nameless call: the writer drops it with a warning.
                    Err(_) => out.fragment(
                        index,
                        CallFragment {
                            id: Some(id),
                            name: None,
                            arguments: None,
                        },
                    )?,
                }
                // A whole reply states the input on the block; a stream
                // streams it, and the fragments win.
                return out.announce(index, input);
            }
            _ => {
                self.inputs.insert(index, (false, String::new()));
                (Block::Opaque { replay: true }, String::new())
            }
        };
        out.open(index, opened, Value::Object(block))?;
        out.push(index, &text)
    }

    /// Apply a delta to the open block at `index`: text and reasoning grow
    /// both the block and its item, input JSON is assembled, a citation is
    /// appended, and any other delta merges into the item by key.
    fn delta(
        &mut self,
        index: usize,
        delta: Map<String, Value>,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        // The fragment a block is built from must be text.
        let fragment = |key: &str| {
            delta.get(key).and_then(Value::as_str).ok_or_else(|| {
                ProviderError::Response(format!(
                    "Anthropic `{}` carries no string `{key}`",
                    str_of(&delta, "type")
                ))
            })
        };
        match str_of(&delta, "type") {
            "text_delta" => {
                out.push(index, fragment("text")?)?;
                out.merge(index, &delta)
            }
            "thinking_delta" => {
                out.push(index, fragment("thinking")?)?;
                out.merge(index, &delta)
            }
            "input_json_delta" => {
                let fragment = fragment("partial_json")?;
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
                let citation = delta.get("citation").cloned().unwrap_or_default();
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
            _ => out.merge(index, &delta),
        }
    }

    /// End the block at `index`, as `content_block_stop` states it complete:
    /// its streamed input set on its item, which becomes the block's native.
    /// Input that is not a JSON object leaves the item unusable, so the
    /// block closes without one and replays from its canonical fields.
    fn stop(&mut self, index: usize, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
        let complete = match self.inputs.remove(&index) {
            None => true,
            Some((_, json)) if json.is_empty() => {
                let mut object = true;
                out.edit(index, |item| {
                    object = item.get("input").is_none_or(Value::is_object);
                })?;
                object
            }
            Some((_, json)) => match crate::json_utils::parse_tool_arguments(&json) {
                Ok(input) if input.is_object() => {
                    out.edit(index, |item| {
                        if let Some(item) = item.as_object_mut() {
                            item.insert("input".to_owned(), input);
                        }
                    })?;
                    true
                }
                Ok(_) | Err(_) => false,
            },
        };
        if complete {
            out.finish(index)
        } else {
            out.close(index)
        }
    }

    /// Note the metadata a whole message or `message_start` states.
    fn metadata(&mut self, message: &Map<String, Value>) {
        self.start = Counts::of(message.get("usage"));
        self.message_id = string_of(message, "id");
        self.response_model = string_of(message, "model");
        self.note_container(message.get("container"));
    }

    fn note_container(&mut self, container: Option<&Value>) {
        if let Some(container) = container.filter(|container| !container.is_null()) {
            self.container = Some(container.clone());
        }
    }

    /// End the reply with Anthropic's terminal record.
    fn end(
        &self,
        usage: &Counts,
        stop_reason: Option<&str>,
        details: Option<&Value>,
        mut out: Out<'_, Completion>,
    ) -> Flow {
        if let Some(container) = &self.container {
            out.message_native(serde_json::json!({ "container": container }));
        }
        let (reason, error) = match stop_reason {
            Some(reason) => {
                let (reason, error) = finish_of(reason, details);
                (Some(reason), error)
            }
            None => (None, None),
        };
        out.end(Finish {
            usage: usage.usage(),
            reason,
            response_id: self.message_id.clone(),
            model: self.response_model.clone(),
            error,
        })
    }

    /// A whole message, written block by block through the calls a stream
    /// makes, then ended. Empty content is a turn like any other, as it is
    /// streamed (pi's rule): the stop reason decides how it ends.
    fn whole(
        &mut self,
        message: Map<String, Value>,
        mut out: Out<'_, Completion>,
    ) -> Result<Flow, ProviderError> {
        self.metadata(&message);
        let stop_reason = message.get("stop_reason").and_then(Value::as_str);
        let content = match message.get("content") {
            Some(Value::Array(content)) => content.as_slice(),
            _ => &[],
        };
        for (index, block) in content.iter().enumerate() {
            let Some(block) = block
                .as_object()
                .filter(|block| block.get("type").is_some_and(Value::is_string))
            else {
                return Err(ProviderError::Response(format!(
                    "Anthropic content block {index} has no string `type`"
                )));
            };
            self.start(index, block.clone(), &mut out)?;
            self.stop(index, &mut out)?;
        }
        let usage = self.start.clone();
        Ok(self.end(&usage, stop_reason, message.get("stop_details"), out))
    }

    /// The stream's terminal `message_delta`: its counters, falling back to
    /// `message_start`'s for those it omits.
    fn terminal(&self, usage: Option<&Value>) -> Counts {
        let terminal = Counts::of(usage);
        Counts {
            // Zero-as-missing is a gateway heuristic for the input count
            // only, not a rule for cache counts.
            input: terminal
                .input
                .filter(|tokens| *tokens > 0)
                .or(self.start.input),
            output: terminal.output,
            cache_read: terminal.cache_read.or(self.start.cache_read),
            cache_creation: terminal.cache_creation.or(self.start.cache_creation),
            cache_creation_split: terminal
                .cache_creation_split
                .or_else(|| self.start.cache_creation_split.clone()),
            thinking: terminal.thinking,
        }
    }
}

impl<'id> Decoder<'id, Completion> for MessagesDecoder {
    type Event = MessagesEvent;

    fn classify(&self, frame: WireFrame) -> WireEvent<MessagesEvent> {
        let data = frame.as_str();
        wire::classify_tagged_frame::<Map<String, Value>>(&data, "type", |event_type| {
            KNOWN_EVENT_TYPES.contains(&event_type)
        })
        .map(|fields| {
            // The one event whose payload leaves this crate as bytes rather
            // than as decoded fields, so it is captured where the frame is
            // still in hand.
            let raw = (str_of(&fields, "type") == "error").then(|| data.to_string());
            MessagesEvent { fields, raw }
        })
    }

    fn decode(
        &mut self,
        event: MessagesEvent,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        match event.kind() {
            "message" => return self.whole(event.fields, out),
            // A `message_start` without a message body (a Bedrock-compatible
            // gateway sends one) is a no-op.
            "message_start" => {
                if let Some(Value::Object(message)) = event.fields.get("message") {
                    self.metadata(message);
                }
            }
            "content_block_start" => {
                let block = event.item("content_block")?;
                self.start(event.index()?, block, &mut out)?;
            }
            "content_block_delta" => {
                let delta = event.item("delta")?;
                self.delta(event.index()?, delta, &mut out)?;
            }
            "content_block_stop" => self.stop(event.index()?, &mut out)?,
            "message_delta" => {
                let delta = event.fields.get("delta");
                self.note_container(delta.and_then(|delta| delta.get("container")));
                // Only a `message_delta` carrying a stop reason is the
                // provider's end; without one it is a no-op.
                let Some(reason) = delta
                    .and_then(|delta| delta.get("stop_reason"))
                    .and_then(Value::as_str)
                else {
                    return Ok(Flow::More);
                };
                let usage = self.terminal(event.fields.get("usage"));
                out.raw(serde_json::to_value(StreamingCompletionResponse {
                    usage: usage.partial(),
                    stop_reason: Some(reason.to_owned()),
                    // Rides the same `message_delta` as the stop reason:
                    // `message_start` always opens with `null`.
                    stop_sequence: delta
                        .and_then(|delta| delta.get("stop_sequence"))
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                    message_id: self.message_id.clone(),
                    model: self.response_model.clone(),
                })?);
                let details = delta.and_then(|delta| delta.get("stop_details"));
                return Ok(self.end(&usage, Some(reason), details, out));
            }
            // Preserve the complete error envelope rather than re-encode
            // modeled fields.
            "error" => {
                return Err(ProviderError::from_provider_body(
                    event.raw.unwrap_or_default(),
                ));
            }
            // `message_stop`, `ping`, and nothing else: the classifier
            // passes only the listed tags.
            _ => {}
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
        let Ok(Value::Object(payload)) = serde_json::from_slice::<Value>(payload) else {
            return;
        };
        let message = payload.get("message").and_then(Value::as_object);
        let fields = message.unwrap_or(&payload);
        // Anthropic reports the prompt on `message_start` and the answer's
        // running total on each `message_delta`: each is a snapshot of what it
        // knows, never a sum.
        if let Some(usage) = payload.get("usage").or_else(|| fields.get("usage")) {
            let counts = Counts::of(Some(usage));
            sink.emit(AdapterEvent::Usage {
                usage: AdapterUsage {
                    input_tokens: counts.input,
                    output_tokens: counts.output,
                    total_tokens: None,
                    cached_input_tokens: counts.cache_read,
                    reasoning_tokens: counts.thinking,
                    tool_input_tokens: None,
                },
            });
        }
        let stop_reason = string_of(fields, "stop_reason").or_else(|| {
            payload
                .get("delta")
                .and_then(Value::as_object)
                .and_then(|delta| string_of(delta, "stop_reason"))
        });
        let verdict = AdapterVerdict {
            finish_reason: stop_reason.map(|v| sink.scrub(&v)),
            block_reason: None,
            detail: None,
            model: string_of(fields, "model").map(|v| sink.scrub(&v)),
        };
        let response_id = string_of(fields, "id").map(|v| sink.scrub(&v));
        sink.provider(verdict, response_id);
        if let Some(error) = payload
            .get("error")
            .and_then(|error| ObservedError::deserialize(error).ok())
        {
            error.emit(sink);
        }
    }
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
