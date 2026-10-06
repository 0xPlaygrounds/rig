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
        let part = |key| payload.get(key).unwrap_or(&Value::Null);
        match kind {
            "contentBlockStart" => self.start(out, index, part("start"))?,
            "contentBlockDelta" => self.delta(out, index, part("delta"))?,
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
        start: &Value,
    ) -> Result<(), ProviderError> {
        let Some((kind, body)) = member(start) else {
            return Ok(());
        };
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
                out.edit(index, |slot| start.clone_into(slot))?;
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
                out.open(index, Block::Opaque { replay }, start.clone())?;
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
    /// and it merges into the block's JSON.
    fn delta(
        &mut self,
        out: &mut Out<'_, Completion>,
        index: usize,
        delta: &Value,
    ) -> Result<(), ProviderError> {
        let Some((kind, body)) = member(delta) else {
            return Ok(());
        };
        let redacted = body.get("redactedContent");
        if !self.open.contains_key(&index) {
            let (block, open) = match kind {
                "text" | "citation" => (Block::Text, Open::Text),
                "reasoningContent" => {
                    let redacted = redacted.is_some();
                    (Block::Reasoning { redacted }, Open::Reasoning)
                }
                "toolUse" => {
                    self.start(out, index, &json!({ "toolUse": {} }))?;
                    return self.delta(out, index, delta);
                }
                kind => {
                    skip(kind);
                    return Ok(());
                }
            };
            out.open(index, block, Value::Null)?;
            self.open.insert(index, open);
        }
        // Redacted bytes are kept as their chunks and encoded once whole:
        // base64 of each chunk would not concatenate.
        let open = self.open.get(&index);
        if let (Some(text), Some(Open::Text | Open::Reasoning)) =
            (body.as_str().or_else(|| body.str("text")), open)
        {
            out.push(index, text)?;
        }
        let (pointer, merged) = match (kind, open) {
            ("text", Some(Open::Text)) => ("", json!({ "text": body })),
            ("citation", Some(Open::Text)) => ("", json!({ "citations": [body] })),
            ("reasoningContent", Some(Open::Reasoning)) => match redacted {
                Some(chunk) => ("", json!({ "redactedContent": [chunk] })),
                None => ("", body.clone()),
            },
            ("toolUse", Some(open @ (Open::Call | Open::Hosted))) => {
                if matches!(open, Open::Call) {
                    let arguments = body.str("input");
                    let fragment = CallFragment {
                        id: None,
                        name: None,
                        arguments,
                    };
                    out.fragment(Some(index), fragment)?;
                }
                ("/toolUse", body.clone())
            }
            ("toolResult", Some(Open::Opaque)) => ("/toolResult", json!({ "content": body })),
            ("image", Some(Open::Image)) => {
                if let Some(data) = body.at("/source/bytes").and_then(Value::as_str) {
                    out.push(index, data)?;
                }
                return Ok(());
            }
            (kind, _) => {
                skip(kind);
                return Ok(());
            }
        };
        out.edit(index, |item| {
            if let (Some(at), Value::Object(merged)) = (item.pointer_mut(pointer), &merged) {
                merge(at, merged);
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
        let call = matches!(open, Open::Call | Open::Hosted);
        out.edit(index, |item| {
            let built = match open {
                Open::Text => (!item.arr("citations").is_empty()).then(|| {
                    let content = [json!({ "text": item.get("text") })];
                    json!({ "citationsContent": { "content": content, "citations": item.get("citations") } })
                }),
                // Unsigned reasoning keeps no item: it is rebuilt from its
                // text, as text for a family that rejects unsigned reasoning.
                Open::Reasoning => match (item.get("redactedContent"), item.str("signature")) {
                    (Some(chunks), _) => {
                        let bytes: Vec<u8> = chunks
                            .as_array()
                            .into_iter()
                            .flatten()
                            .filter_map(|chunk| BASE64_STANDARD.decode(chunk.as_str()?).ok())
                            .flatten()
                            .collect();
                        let redacted = BASE64_STANDARD.encode(bytes);
                        Some(json!({ "reasoningContent": { "redactedContent": redacted } }))
                    }
                    (None, Some(signature)) => {
                        let text = item.str("text").unwrap_or_default();
                        let reasoning = json!({ "text": text, "signature": signature });
                        Some(json!({ "reasoningContent": { "reasoningText": reasoning } }))
                    }
                    (None, None) => None,
                },
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
                    input.and_then(|input| {
                        let mut call = item.get("toolUse")?.clone();
                        merge(&mut call, &Map::from_iter([("input".to_owned(), input)]));
                        Some(json!({ "toolUse": call }))
                    })
                }
                Open::Image => None,
                Open::Opaque => Some(item.clone()),
            };
            *item = match (built, whole) {
                (Some(_), Some(whole)) if !call => whole.clone(),
                (built, _) => built.unwrap_or_default(),
            };
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
            let Some((kind, body)) = member(block) else {
                continue;
            };
            // A started block's streamed field arrives as its delta.
            let (started, deltas) = match kind {
                "text" => (None, vec![json!({ "text": body })]),
                "citationsContent" => {
                    let text = body
                        .arr("content")
                        .iter()
                        .map(|part| json!({ "text": part.get("text") }));
                    let cited = body
                        .arr("citations")
                        .iter()
                        .map(|citation| json!({ "citation": citation }));
                    (None, text.chain(cited).collect())
                }
                "reasoningContent" => {
                    let reasoning = body.get("reasoningText").unwrap_or(body);
                    (None, vec![json!({ "reasoningContent": reasoning })])
                }
                "toolUse" => {
                    let input = body.get("input").map(Value::to_string);
                    (
                        Some("input"),
                        vec![json!({ "toolUse": { "input": input } })],
                    )
                }
                "toolResult" => (
                    Some("content"),
                    vec![json!({ "toolResult": body.get("content") })],
                ),
                "image" => (
                    Some("source"),
                    vec![json!({ "image": { "source": body.get("source") } })],
                ),
                _ => (Some(""), Vec::new()),
            };
            if let Some(field) = started {
                let mut opened = body.clone();
                if let Value::Object(fields) = &mut opened {
                    fields.shift_remove(field);
                }
                self.start(&mut out, index, &json!({ kind: opened }))?;
            }
            for delta in &deltas {
                self.delta(&mut out, index, delta)?;
            }
            self.stop(&mut out, index, Some(block))?;
        }
        self.reason = document.str("stopReason").map(str::to_owned);
        let end = json!({ "metadata": { "usage": document.get("usage") } });
        Ok(match self.event(&end, &mut out)? {
            Some(end) => out.end(end),
            None => Flow::More,
        })
    }
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
    Usage::new()
        .input_tokens(input)
        .output_tokens(output)
        .total_tokens(input.saturating_add(output))
        .cached_input_tokens(cache_read)
        .cache_creation_input_tokens(cache_write)
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
