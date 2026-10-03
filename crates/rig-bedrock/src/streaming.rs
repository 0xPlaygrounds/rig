//! The Converse reply decoder. It reads the JSON Bedrock sent through
//! [`Lenient`]: a stream's events one by one, and a whole reply as the
//! events that would have streamed it, so both fold into the same turn.
//!
//! Every Converse content block becomes a block, its Converse JSON assembled
//! as its deltas arrive. Signed or redacted reasoning, cited text, a client
//! call with object input, and every block Rig has no canonical form for
//! keep that JSON as the provider item, set only when the block is
//! complete: at its `contentBlockStop`, or in a whole reply, where the item
//! is the block as Bedrock sent it. A hosted tool's typed `toolUse` and its
//! `toolResult` are opaque items that replay to the same model together,
//! never a call Rig runs.

use std::collections::BTreeMap;

use base64::{Engine, prelude::BASE64_STANDARD};
use rig_core::completion::{FinishReason, Usage};
use rig_core::error::ProviderError;
use rig_core::json_utils::Lenient;
use rig_core::message::{DocumentSourceKind, Image, ImageMediaType};
use rig_core::operation::{Block, CallFragment, Completion, Finish, merge};
use rig_core::wire::{Flow, Out, WireEvent};
use serde_json::{Map, Value, json};

use crate::completion::ConverseFrame;
use crate::types::errors;

/// What an open block is. The writer holds its JSON.
enum Open {
    Text,
    Reasoning,
    Call,
    Hosted,
    Image,
    Opaque,
}

/// One Converse reply's state: a whole reply or a stream of events.
#[derive(Default)]
pub struct StreamState {
    open: BTreeMap<usize, Open>,
    reason: Option<String>,
    /// The stream's message-level events, by type: the response's `raw`.
    raw: Map<String, Value>,
}

/// The one key of a Converse union and its value.
fn member(value: &Value) -> Option<(&str, &Value)> {
    let (key, value) = value.as_object()?.iter().next()?;
    Some((key.as_str(), value))
}

impl StreamState {
    /// Write one event in delivery order; the reply's end at `metadata`.
    fn event(
        &mut self,
        event: &Value,
        out: &mut Out<'_, Completion>,
    ) -> Result<Option<Finish>, ProviderError> {
        let Some((kind, payload)) = member(event) else {
            return Ok(None);
        };
        let index = payload
            .u64("contentBlockIndex")
            .and_then(|index| usize::try_from(index).ok())
            .unwrap_or(0);
        match kind {
            "contentBlockStart" => self.start(out, index, payload.get("start"))?,
            "contentBlockDelta" => self.delta(out, index, payload.get("delta"))?,
            "contentBlockStop" => self.stop(out, index, None)?,
            "messageStop" => self.reason = payload.str("stopReason").map(str::to_owned),
            // The writer closes every block still open as incomplete.
            "metadata" => {
                return Ok(Some(Finish {
                    usage: payload.at("/usage").map(usage).unwrap_or_default(),
                    reason: self.reason.as_deref().map(finish_reason),
                    ..Finish::default()
                }));
            }
            "messageStart" => {}
            kind if kind.ends_with("Exception") => return Err(errors::exception(kind, payload)),
            kind => skip(kind),
        }
        Ok(None)
    }

    /// Converse starts tool-use, tool-result and image blocks, and any
    /// block it adds later.
    fn start(
        &mut self,
        out: &mut Out<'_, Completion>,
        index: usize,
        start: Option<&Value>,
    ) -> Result<(), ProviderError> {
        let Some((kind, body)) = start.and_then(member) else {
            return Ok(());
        };
        let item = json!({ kind: body });
        let open = match kind {
            // A call that never names its tool is dropped at its end, since
            // nothing can answer it.
            "toolUse" if body.str("type").is_none() => {
                let fragment = CallFragment {
                    id: body.str("toolUseId"),
                    name: body.str("name"),
                    arguments: None,
                };
                out.fragment(Some(index), fragment)?;
                out.edit(index, |slot| *slot = item)?;
                Open::Call
            }
            "image" if let Some(media_type) = media_type(body.str("format")) => {
                let image = Image {
                    data: DocumentSourceKind::Base64(String::new()),
                    media_type: Some(media_type),
                    ..Image::default()
                };
                out.open(index, Block::Image(image), Value::Null)?;
                Open::Image
            }
            // An image in a format Rig does not name keeps its JSON and does
            // not replay.
            kind => {
                let replay = kind != "image";
                out.open(index, Block::Opaque { replay }, item)?;
                if kind == "toolUse" {
                    Open::Hosted
                } else {
                    Open::Opaque
                }
            }
        };
        self.open.insert(index, open);
        Ok(())
    }

    /// One delta: its text, reasoning or arguments go out as they arrive,
    /// and it merges into the block's Converse JSON.
    fn delta(
        &mut self,
        out: &mut Out<'_, Completion>,
        index: usize,
        delta: Option<&Value>,
    ) -> Result<(), ProviderError> {
        let Some((kind, body)) = delta.and_then(member) else {
            return Ok(());
        };
        let redacted = body.get("redactedContent");
        if !self.open.contains_key(&index) {
            let (block, item, open) = match kind {
                "text" | "citation" => (
                    Block::Text,
                    json!({ "citationsContent": { "content": [{ "text": "" }], "citations": [] } }),
                    Open::Text,
                ),
                "reasoningContent" => (
                    Block::Reasoning {
                        redacted: redacted.is_some(),
                    },
                    json!({ "reasoningContent": { "reasoningText": { "text": "" } } }),
                    Open::Reasoning,
                ),
                "toolUse" => {
                    self.start(out, index, Some(&json!({ "toolUse": {} })))?;
                    return self.delta(out, index, delta);
                }
                kind => {
                    skip(kind);
                    return Ok(());
                }
            };
            out.open(index, block, item)?;
            self.open.insert(index, open);
        }
        let text = body.as_str().or_else(|| body.str("text"));
        match (kind, self.open.get(&index), text) {
            ("text", Some(Open::Text), Some(text))
            | ("reasoningContent", Some(Open::Reasoning), Some(text)) => out.push(index, text)?,
            ("toolUse", Some(Open::Call), _) => {
                let arguments = body.str("input");
                let fragment = CallFragment {
                    id: None,
                    name: None,
                    arguments,
                };
                out.fragment(Some(index), fragment)?;
            }
            ("image", Some(Open::Image), _) => {
                if let Some(data) = body.at("/source/bytes").and_then(Value::as_str) {
                    out.push(index, data)?;
                }
            }
            _ => {}
        }
        // Redacted bytes are kept as their chunks and encoded once whole:
        // base64 of each chunk would not concatenate.
        let (pointer, delta) = match kind {
            "text" => ("/citationsContent/content/0", json!({ "text": body })),
            "citation" => ("/citationsContent", json!({ "citations": [body] })),
            "reasoningContent" if redacted.is_some() => (
                "/reasoningContent",
                json!({ "redactedContent": [redacted] }),
            ),
            "reasoningContent" => ("/reasoningContent/reasoningText", body.clone()),
            "toolUse" => ("/toolUse", body.clone()),
            "toolResult" => ("/toolResult", json!({ "content": body })),
            "image" => return Ok(()),
            kind => {
                skip(kind);
                return Ok(());
            }
        };
        out.edit(index, |item| {
            if let (Some(at), Value::Object(delta)) = (item.pointer_mut(pointer), &delta) {
                merge(at, delta);
            }
        })
    }

    /// The block at `index` stopped: it becomes whole, with `whole` as its
    /// item when it came in a whole reply.
    fn stop(
        &mut self,
        out: &mut Out<'_, Completion>,
        index: usize,
        whole: Option<&Value>,
    ) -> Result<(), ProviderError> {
        let Some(open) = self.open.remove(&index) else {
            return Ok(());
        };
        out.edit(index, |item| {
            let kept = match open {
                Open::Text => item
                    .at("/citationsContent/citations")
                    .and_then(Value::as_array)
                    .is_some_and(|citations| !citations.is_empty()),
                // Unsigned reasoning keeps no item: it is rebuilt from its
                // text, as text for a family that rejects unsigned reasoning.
                Open::Reasoning => {
                    if let Some(Value::Object(reasoning)) = item.get_mut("reasoningContent")
                        && let Some(Value::Array(chunks)) = reasoning.get("redactedContent")
                    {
                        let bytes: Vec<u8> = chunks
                            .iter()
                            .filter_map(|chunk| BASE64_STANDARD.decode(chunk.as_str()?).ok())
                            .flatten()
                            .collect();
                        reasoning.clear();
                        reasoning.insert(
                            "redactedContent".to_owned(),
                            json!(BASE64_STANDARD.encode(bytes)),
                        );
                    }
                    item.at("/reasoningContent/reasoningText/signature")
                        .is_some_and(Value::is_string)
                        || item.at("/reasoningContent/redactedContent").is_some()
                }
                // Converse takes a JSON object as a tool's input; a call's
                // input is the one its arguments were read from.
                Open::Call | Open::Hosted => {
                    let input = item
                        .at("/toolUse/input")
                        .and_then(Value::as_str)
                        .filter(|input| !input.trim().is_empty())
                        .unwrap_or("{}");
                    let input = serde_json::from_str::<Value>(input)
                        .ok()
                        .filter(Value::is_object)
                        .or_else(|| matches!(open, Open::Hosted).then(|| json!({})));
                    let kept = input.is_some();
                    if let (Some(Value::Object(call)), Some(input)) =
                        (item.get_mut("toolUse"), input)
                    {
                        call.insert("input".to_owned(), input);
                    }
                    kept
                }
                Open::Image => false,
                Open::Opaque => true,
            };
            match (kept, whole) {
                (false, _) => *item = Value::Null,
                (true, Some(whole)) if !matches!(open, Open::Call | Open::Hosted) => {
                    whole.clone_into(item);
                }
                (true, _) => {}
            }
        })?;
        out.finish(index)
    }

    /// A whole reply, written as the events that would have streamed it.
    fn whole(
        &mut self,
        document: &Value,
        mut out: Out<'_, Completion>,
    ) -> Result<Flow, ProviderError> {
        let content = document
            .at("/output/message")
            .map_or(&[][..], |message| message.arr("content"));
        for (index, block) in content.iter().enumerate() {
            for event in restated(index, block) {
                self.event(&event, &mut out)?;
            }
            self.stop(&mut out, index, Some(block))?;
        }
        self.reason = document.str("stopReason").map(str::to_owned);
        let end = json!({ "metadata": { "usage": document.get("usage") } });
        match self.event(&end, &mut out)? {
            Some(end) => Ok(out.end(end)),
            None => Ok(Flow::More),
        }
    }
}

/// The stream events of the whole block at `index`, its stop aside.
fn restated(index: usize, block: &Value) -> Vec<Value> {
    let delta = |delta: Value| json!({ "contentBlockDelta": { "contentBlockIndex": index, "delta": delta } });
    let start = |start: Value| json!({ "contentBlockStart": { "contentBlockIndex": index, "start": start } });
    let Some((kind, body)) = member(block) else {
        return Vec::new();
    };
    // A started block's streamed field arrives as its delta.
    let (field, streamed) = match kind {
        "text" => return vec![delta(json!({ "text": body }))],
        "citationsContent" => {
            let content = body.arr("content").iter();
            let text = content.map(|part| delta(json!({ "text": part.get("text") })));
            let citations = body.arr("citations").iter();
            let citations = citations.map(|citation| delta(json!({ "citation": citation })));
            return text.chain(citations).collect();
        }
        "reasoningContent" => {
            let body = body.get("reasoningText").unwrap_or(body);
            return vec![delta(json!({ "reasoningContent": body }))];
        }
        "toolUse" => {
            let input = body.get("input").map(Value::to_string);
            ("input", json!({ "toolUse": { "input": input } }))
        }
        "toolResult" => ("content", json!({ "toolResult": body.get("content") })),
        "image" => (
            "source",
            json!({ "image": { "source": body.get("source") } }),
        ),
        _ => return vec![start(block.clone())],
    };
    let mut opened = body.clone();
    if let Value::Object(fields) = &mut opened {
        fields.shift_remove(field);
    }
    vec![start(json!({ kind: opened })), delta(streamed)]
}

/// The image type a Converse image `format` names.
fn media_type(format: Option<&str>) -> Option<ImageMediaType> {
    match format? {
        "gif" => Some(ImageMediaType::GIF),
        "jpeg" => Some(ImageMediaType::JPEG),
        "png" => Some(ImageMediaType::PNG),
        "webp" => Some(ImageMediaType::WEBP),
        _ => None,
    }
}

fn skip(item: &str) {
    tracing::warn!(item, "a Converse item this crate does not model");
}

/// Every stop reason Converse documents. An end of turn or a stop sequence
/// is `Stop`, the token limit and an exceeded context window are `Length`,
/// guardrail and content-filter interventions are `ContentFilter`, and any
/// other reason keeps its wire spelling in `Other`, which fails the turn.
fn finish_reason(reason: &str) -> FinishReason {
    match reason {
        "end_turn" | "stop_sequence" => FinishReason::Stop,
        "max_tokens" | "model_context_window_exceeded" => FinishReason::Length,
        "tool_use" => FinishReason::ToolCalls,
        "content_filtered" | "guardrail_intervened" => FinishReason::ContentFilter,
        other => FinishReason::Other(other.to_owned()),
    }
}

/// Rig's input is Bedrock's `inputTokens` plus its cache reads and writes,
/// and its total input plus output, which is `totalTokens`.
fn usage(usage: &Value) -> Usage {
    let cache_read = usage.u64("cacheReadInputTokens");
    let cache_write = usage.u64("cacheWriteInputTokens");
    let input = [usage.u64("inputTokens"), cache_read, cache_write]
        .into_iter()
        .flatten()
        .fold(0, u64::saturating_add);
    let output = usage.u64("outputTokens").unwrap_or(0);
    Usage {
        input_tokens: Some(input),
        output_tokens: Some(output),
        total_tokens: Some(input.saturating_add(output)),
        cached_input_tokens: cache_read,
        cache_creation_input_tokens: cache_write,
        ..Usage::default()
    }
}

impl<'id> rig_core::wire::Decoder<'id, Completion, ConverseFrame> for StreamState {
    type Event = ConverseFrame;

    fn classify(&self, frame: ConverseFrame) -> WireEvent<Self::Event> {
        WireEvent::Known(frame)
    }

    /// EOF without Bedrock's `metadata` event is truncation.
    fn decode(
        &mut self,
        frame: ConverseFrame,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        let event = match frame {
            ConverseFrame::Whole(document) => return self.whole(&document, out),
            ConverseFrame::Event(event) => event,
        };
        if let Some((kind @ ("messageStart" | "messageStop" | "metadata"), payload)) =
            member(&event)
        {
            self.raw.insert(kind.to_owned(), payload.clone());
        }
        let Some(end) = self.event(&event, &mut out)? else {
            return Ok(Flow::More);
        };
        if !self.raw.is_empty() {
            out.raw(Value::Object(std::mem::take(&mut self.raw)));
        }
        Ok(out.end(end))
    }
}

#[cfg(test)]
pub(crate) mod tests;
