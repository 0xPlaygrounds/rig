//! The Messages reply decoder, for a whole message and for the event stream
//! alike. Each content block becomes one block, in wire order, whose
//! provider item is the block as the provider states it complete, and only
//! when the wire would take that item back.
//!
//! ```
//! use rig_core::providers::anthropic::streaming::MessagesDecoder;
//!
//! let decoder = MessagesDecoder::new(false);
//! # let _ = decoder;
//! ```

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value, json};

use super::completion::object;
use crate::completion::FinishReason;
use crate::error::ProviderError;
use crate::json_utils::Lenient;
use crate::message::{CallId, DocumentRange, Source, SourceLocation, ToolName};
use crate::observe::ObservedError;
use crate::operation::{Block, CallFragment, Completion, Finish};
use crate::providers::internal::wire;
use crate::wire::{
    AdapterVerdict, Decoder, Flow, ObservationSink, Out, WireCitation, WireEvent, WireFrame,
};

/// Recognized Messages event tags; any other tag classifies as unknown.
/// `message` is the whole message a unary reply is.
const KNOWN_EVENT_TYPES: &[&str] = &[
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
    /// The event as sent.
    pub fields: Value,
    /// The frame's text when it is an `error` event, whose envelope is
    /// reported verbatim.
    raw: Option<String>,
}

impl MessagesEvent {
    fn kind(&self) -> &str {
        self.fields.str("type").unwrap_or_default()
    }

    /// The content block the event addresses.
    fn index(&self) -> Result<usize, ProviderError> {
        self.fields
            .u64("index")
            .and_then(|index| usize::try_from(index).ok())
            .ok_or_else(|| {
                ProviderError::Response(format!("Anthropic `{}` names no block index", self.kind()))
            })
    }

    /// The block or delta under `key`, which must name its `type`.
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

/// Anthropic's usage counters, read leniently.
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
        let count = |pointer: &str| usage?.at(pointer)?.as_u64_lenient();
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
            cost: None,
        }
    }
}

/// How a Messages `stop_reason` ends the turn, and the error a refusal
/// reports. A reason Anthropic does not document is
/// [`FinishReason::Other`], which fails the turn.
fn finish_of(reason: &str, details: Option<&Value>) -> (Option<FinishReason>, Option<String>) {
    let reason = match reason {
        // `pause_turn` is a server-tool loop that stopped at its limit: the
        // turn is replayed as it is to resume it (pi's rule).
        "end_turn" | "stop_sequence" | "pause_turn" => FinishReason::Stop,
        "max_tokens" | "model_context_window_exceeded" => FinishReason::Length,
        "tool_use" => FinishReason::ToolCalls,
        // A refusal fails the turn with its explanation (pi's rule).
        "refusal" => {
            let explanation = details
                .and_then(|details| details.str("explanation"))
                .filter(|explanation| !explanation.is_empty())
                .unwrap_or("The model refused to complete the request");
            return (
                Some(FinishReason::ContentFilter),
                Some(explanation.to_owned()),
            );
        }
        other => FinishReason::Other(other.to_owned()),
    };
    (Some(reason), None)
}

/// One Messages citation as a whole-block [`WireCitation`]: Anthropic
/// cites each text block as a whole. A location kind rig does not know, or
/// one without the field that names its source, is `None` and stays only in
/// the block's native item, as does `encrypted_index`. Document ranges end
/// exclusive, pages count from 1, characters and blocks from 0.
fn citation_of(citation: &Value) -> Option<WireCitation> {
    let number = |key: &str| citation.u64(key).and_then(|n| u32::try_from(n).ok());
    let range = |start: &str, end: &str| Some(number(start)?..number(end)?);
    let location = match citation.str("type")? {
        "char_location" => SourceLocation::Document {
            index: number("document_index"),
            id: citation.str("file_id").map(str::to_owned),
            within: citation
                .u64("start_char_index")
                .zip(citation.u64("end_char_index"))
                .map(|(start, end)| DocumentRange::Chars(start..end)),
        },
        "page_location" => SourceLocation::Document {
            index: number("document_index"),
            id: citation.str("file_id").map(str::to_owned),
            within: range("start_page_number", "end_page_number").map(DocumentRange::Pages),
        },
        "content_block_location" => SourceLocation::Document {
            index: number("document_index"),
            id: citation.str("file_id").map(str::to_owned),
            within: range("start_block_index", "end_block_index").map(DocumentRange::Blocks),
        },
        "search_result_location" => SourceLocation::SearchResult {
            index: number("search_result_index")?,
            source: citation.str("source")?.to_owned(),
            blocks: range("start_block_index", "end_block_index"),
        },
        "web_search_result_location" => SourceLocation::Url {
            url: citation.str("url")?.to_owned(),
        },
        _ => return None,
    };
    let mut source = Source::new(location);
    if let Some(title) = citation
        .str("document_title")
        .or_else(|| citation.str("title"))
    {
        source = source.title(title);
    }
    if let Some(cited) = citation.str("cited_text") {
        source = source.cited_text(cited);
    }
    Some(WireCitation::new(None, vec![source]))
}

/// What an open content block is, for the checks its end makes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Text,
    Thinking,
    Redacted,
    Call,
    Opaque,
}

/// Decodes Messages replies, a whole message or a stream of events.
/// `content_block_stop` states a block complete; a stop reason states every
/// block still open complete, but a call whose input is not yet JSON.
#[derive(Debug, Default)]
pub struct MessagesDecoder {
    /// Each open block's kind and the input JSON streamed to it.
    open: BTreeMap<usize, (Kind, String)>,
    /// Every index a block opened at, closed or not.
    started: BTreeSet<usize>,
    /// Whether a block other than a leading `fallback` marker opened.
    opened: bool,
    /// Whether the dialect takes thinking back without a signature.
    unsigned_thinking: bool,
    /// The counters `message_start` reported, for a terminal `message_delta`
    /// that does not repeat them.
    start: Counts,
    message_id: Option<String>,
    response_model: Option<String>,
    container: Option<Value>,
}

impl MessagesDecoder {
    /// A fresh decoder for one reply. `unsigned_thinking` is whether the
    /// dialect takes thinking back without a signature.
    pub fn new(unsigned_thinking: bool) -> Self {
        Self {
            unsigned_thinking,
            ..Self::default()
        }
    }

    /// Open the content block at `index` as the provider states it.
    fn start(
        &mut self,
        index: usize,
        block: Map<String, Value>,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        self.started.insert(index);
        let block = Value::Object(block);
        let kind = block.str("type").unwrap_or_default();
        // A leading `fallback` names the model that took over; one after
        // output began is a fallback rig cannot represent (pi's rule).
        if kind == "fallback" {
            if self.opened {
                return Err(ProviderError::Response(
                    "Anthropic performed an unsupported mid-output model fallback".to_owned(),
                ));
            }
            self.open.insert(index, (Kind::Opaque, String::new()));
            return out.open(index, Block::Opaque { replay: false }, block);
        }
        self.opened = true;
        let (opened, kind, text) = match kind {
            "text" => (Block::Text, Kind::Text, block.str("text")),
            "thinking" => (
                Block::Reasoning { redacted: false },
                Kind::Thinking,
                block.str("thinking"),
            ),
            "redacted_thinking" => (Block::Reasoning { redacted: true }, Kind::Redacted, None),
            "tool_use" => {
                self.open.insert(index, (Kind::Call, String::new()));
                let id = block.str("id").unwrap_or_default().to_owned();
                let input = block.get("input").cloned().unwrap_or_default();
                match ToolName::new(block.str("name").unwrap_or_default()) {
                    Ok(name) => {
                        let id = CallId::from_wire(&id);
                        out.open(index, Block::Call { id, name }, block)?;
                    }
                    // A nameless call: the writer drops it with a warning.
                    Err(_) => out.fragment(
                        Some(index),
                        CallFragment {
                            id: Some(&id),
                            ..CallFragment::default()
                        },
                    )?,
                }
                // A whole reply states the input on the block; a stream
                // streams it, and the fragments win.
                return out.announce(index, input);
            }
            _ => (Block::Opaque { replay: true }, Kind::Opaque, None),
        };
        let text = text.unwrap_or_default().to_owned();
        let citations: Vec<WireCitation> = match kind {
            Kind::Text => block
                .arr("citations")
                .iter()
                .filter_map(citation_of)
                .collect(),
            _ => Vec::new(),
        };
        self.open.insert(index, (kind, String::new()));
        out.open(index, opened, block)?;
        for citation in citations {
            out.cite(index, citation);
        }
        out.push(index, &text)
    }

    /// Apply a delta to the open block at `index`: text and reasoning grow
    /// both the block and its item, input JSON is assembled, a citation is
    /// appended to the item and cited, and any other delta merges into the
    /// item by key.
    fn delta(
        &mut self,
        index: usize,
        delta: Map<String, Value>,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        let kind = delta
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        // The fragment a block is built from must be text.
        let fragment = |key: &str| {
            delta.get(key).and_then(Value::as_str).ok_or_else(|| {
                ProviderError::Response(format!("Anthropic `{kind}` carries no string `{key}`"))
            })
        };
        // A gateway that skips `content_block_start` still streams the
        // block's text, so the first delta opens it.
        if !self.started.contains(&index) {
            let opened = match kind {
                "text_delta" => Some(json!({"type": "text", "text": ""})),
                "thinking_delta" => Some(json!({"type": "thinking", "thinking": ""})),
                _ => None,
            };
            if let Some(Value::Object(block)) = opened {
                self.start(index, block, out)?;
            }
        }
        match kind {
            "text_delta" => out.push(index, fragment("text")?)?,
            "thinking_delta" => out.push(index, fragment("thinking")?)?,
            "input_json_delta" => {
                let fragment = fragment("partial_json")?;
                let Some((kind @ (Kind::Call | Kind::Opaque), json)) = self.open.get_mut(&index)
                else {
                    return Err(ProviderError::Response(format!(
                        "Anthropic streamed input to content block {index}, which takes none"
                    )));
                };
                json.push_str(fragment);
                if *kind == Kind::Call {
                    out.push(index, fragment)?;
                }
                return Ok(());
            }
            "citations_delta" => {
                let citation = delta.get("citation").cloned().unwrap_or_default();
                let cited = citation_of(&citation);
                out.edit(index, |item| {
                    if let Some(item) = item.as_object_mut() {
                        match item.get_mut("citations") {
                            Some(Value::Array(citations)) => citations.push(citation),
                            _ => {
                                item.insert("citations".to_owned(), Value::Array(vec![citation]));
                            }
                        }
                    }
                })?;
                if let Some(cited) = cited {
                    out.cite(index, cited);
                }
                return Ok(());
            }
            _ => {}
        }
        // Text and signatures concatenate; `compaction_delta` and kinds
        // rig has never seen land in the item too.
        out.edit(index, |item| {
            crate::operation::completion::merge(item, &delta)
        })
    }

    /// End the block at `index` as stated complete. Its item becomes the
    /// block's native only when the wire takes it back: text that is not
    /// blank, thinking with its signature (unless the dialect takes it
    /// unsigned), redacted thinking with its data, and a call with an
    /// object `input`, set from what streamed. A hosted item whose input is
    /// not an object never completed.
    fn stop(&mut self, index: usize, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
        let Some((kind, json)) = self.open.remove(&index) else {
            return out.finish(index);
        };
        let streamed = (!json.is_empty()).then(|| crate::json_utils::parse_tool_arguments(&json));
        let unsigned = self.unsigned_thinking;
        let mut complete = true;
        out.edit(index, |item| {
            let input = match streamed {
                Some(Ok(parsed)) => Ok(Some(parsed)),
                Some(Err(_)) => Err(()),
                None => Ok(item.get("input").cloned()),
            };
            let kept = match (kind, input) {
                (Kind::Text, _) => !item.str("text").unwrap_or_default().trim().is_empty(),
                (Kind::Thinking, _) => {
                    unsigned || item.str("signature").is_some_and(|s| !s.is_empty())
                }
                (Kind::Redacted, _) => item.str("data").is_some_and(|data| !data.is_empty()),
                (Kind::Opaque, Ok(None)) => true,
                (Kind::Call | Kind::Opaque, Ok(Some(input @ Value::Object(_)))) => {
                    crate::operation::completion::merge(item, &object([("input", Some(input))]));
                    true
                }
                (Kind::Call, Ok(None | Some(Value::Null))) => {
                    crate::operation::completion::merge(
                        item,
                        &object([("input", Some(json!({})))]),
                    );
                    true
                }
                (Kind::Call, _) => false,
                (Kind::Opaque, _) => {
                    complete = false;
                    true
                }
            };
            if !kept {
                *item = Value::Null;
            }
        })?;
        if complete {
            out.finish(index)
        } else {
            out.close(index)
        }
    }

    /// Note the metadata a whole message or `message_start` states.
    fn metadata(&mut self, message: &Value) {
        self.start = Counts::of(message.get("usage"));
        self.message_id = message.str("id").map(str::to_owned);
        self.response_model = message.str("model").map(str::to_owned);
        self.note_container(message.get("container"));
    }

    fn note_container(&mut self, container: Option<&Value>) {
        self.container = container
            .filter(|c| !c.is_null())
            .or(self.container.as_ref())
            .cloned();
    }

    /// End the reply with Anthropic's terminal record. Every block still
    /// open is complete, but a call whose input is not JSON yet: the end of
    /// the reply closes it unfinished. The container the reply ran in is a
    /// last opaque block.
    fn end(
        &mut self,
        usage: &Counts,
        stop_reason: Option<&str>,
        details: Option<&Value>,
        mut out: Out<'_, Completion>,
    ) -> Result<Flow, ProviderError> {
        let open: Vec<usize> = self
            .open
            .iter()
            .filter(|(_, (kind, json))| {
                *kind != Kind::Call
                    || json.is_empty()
                    || crate::json_utils::parse_tool_arguments(json).is_ok_and(|v| v.is_object())
            })
            .map(|(index, _)| *index)
            .collect();
        for index in open {
            self.stop(index, &mut out)?;
        }
        if let Some(container) = &self.container {
            let index = out.fresh_index();
            let item = json!({ "type": "container", "container": container });
            out.whole(index, Block::Opaque { replay: true }, item, "")?;
        }
        let (reason, error) = stop_reason.map_or((None, None), |reason| finish_of(reason, details));
        Ok(out.end(Finish {
            usage: usage.usage(),
            reason,
            response_id: self.message_id.clone(),
            model: self.response_model.clone(),
            error,
        }))
    }

    /// A whole message, written block by block through the calls a stream
    /// makes, then ended. Empty content is a turn like any other, as it is
    /// streamed (pi's rule): the stop reason decides how it ends.
    fn whole(
        &mut self,
        message: Value,
        mut out: Out<'_, Completion>,
    ) -> Result<Flow, ProviderError> {
        self.metadata(&message);
        for (index, block) in message.arr("content").iter().enumerate() {
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
        self.end(
            &usage,
            message.str("stop_reason"),
            message.get("stop_details"),
            out,
        )
    }

    /// The stream's terminal `message_delta` counters, falling back to
    /// `message_start`'s for those it omits.
    fn terminal(&self, usage: Option<&Value>) -> Counts {
        let (terminal, start) = (Counts::of(usage), self.start.clone());
        Counts {
            // Zero-as-missing is a gateway heuristic for the input count
            // only, not a rule for cache counts.
            input: terminal.input.filter(|tokens| *tokens > 0).or(start.input),
            cache_read: terminal.cache_read.or(start.cache_read),
            cache_creation: terminal.cache_creation.or(start.cache_creation),
            cache_creation_split: terminal.cache_creation_split.or(start.cache_creation_split),
            ..terminal
        }
    }
}

impl<'id> Decoder<'id, Completion> for MessagesDecoder {
    type Event = MessagesEvent;

    fn classify(&self, frame: WireFrame) -> WireEvent<MessagesEvent> {
        let data = frame.as_str();
        wire::classify_tagged_frame::<Value>(&data, "type", |tag| KNOWN_EVENT_TYPES.contains(&tag))
            .map(|fields| {
                // The one event whose payload leaves this crate as bytes rather
                // than as decoded fields, so it is captured where the frame is
                // still in hand.
                let raw = (fields.str("type") == Some("error")).then(|| data.to_string());
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
                if let Some(message) = event.fields.get("message").filter(|m| m.is_object()) {
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
                let Some(reason) = delta.and_then(|delta| delta.str("stop_reason")) else {
                    return Ok(Flow::More);
                };
                let usage = self.terminal(event.fields.get("usage"));
                let details = delta.and_then(|delta| delta.get("stop_details"));
                return self.end(&usage, Some(reason), details, out);
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
    /// stop reason, the model, the message id and any error envelope, on the
    /// unary reply and on the stream's `message_start`, `message_delta` and
    /// `error` events.
    pub(crate) fn project(payload: &[u8], sink: &mut ObservationSink<'_>) {
        let Ok(payload) = serde_json::from_slice::<Value>(payload) else {
            return;
        };
        let fields = payload
            .get("message")
            .filter(|m| m.is_object())
            .unwrap_or(&payload);
        let stop_reason = fields
            .str("stop_reason")
            .or_else(|| payload.at("/delta/stop_reason").and_then(Value::as_str));
        let verdict = AdapterVerdict {
            finish_reason: stop_reason.map(|reason| sink.scrub(reason)),
            block_reason: None,
            detail: None,
            model: fields.str("model").map(|model| sink.scrub(model)),
        };
        let response_id = fields.str("id").map(|id| sink.scrub(id));
        sink.provider(verdict, response_id);
        if let Some(error) = payload
            .get("error")
            .and_then(|error| serde::Deserialize::deserialize(error).ok())
        {
            ObservedError::emit(error, sink);
        }
    }
}

pub(crate) mod document;

#[cfg(test)]
mod tests;
