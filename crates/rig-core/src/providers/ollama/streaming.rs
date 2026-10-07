//! The decoder of native Ollama chat replies: a whole `/api/chat` body, or a
//! stream of NDJSON records of the same shape, the last one `done`.
//!
//! A record's `thinking` is reasoning, its `content` text and its
//! `tool_calls` whole calls. A reply with no `thinking` may carry its
//! reasoning as a leading `<think>` block in `content`; that block is split
//! out as reasoning (see [`ChatDecoder`]).
//!
//! ```
//! use rig_core::providers::ollama::streaming::ChatDecoder;
//!
//! let decoder = ChatDecoder::default();
//! # let _ = decoder;
//! ```

use serde::Deserialize;
use serde_json::{Map, Value};

use crate::completion::{FinishReason, Usage};
use crate::error::ProviderError;
use crate::json_utils::Lenient;
use crate::message::{CallId, ToolName};
use crate::observe::ObservedError;
use crate::operation::{Block, Completion, Finish};
use crate::providers::internal::wire;
use crate::wire::{AdapterVerdict, Decoder, Flow, ObservationSink, Out, WireEvent, WireFrame};

/// The keys that make a JSON line a chat record: every record carries
/// `message` or `done`, and an in-band failure carries `error`.
const RECORD_KEYS: &[&str] = &["message", "done", "error"];

/// The tag that opens an inline reasoning block.
const THINK_OPEN: &str = "<think>";
/// The tag that closes it.
const THINK_CLOSE: &str = "</think>";

/// One `/api/chat` record, as the daemon sent it.
#[derive(Debug, Default, Deserialize)]
#[serde(transparent)]
pub struct ChatRecord(pub Map<String, Value>);

/// Decode native Ollama chat replies.
///
/// `thinking` and `content` each continue the block of the record before
/// while the kind stays the same, and each call is a whole block. The
/// reply ends at the record that says `done`; its `done_reason` is the
/// finish reason, `tool_calls` when a `stop` reply called a tool, and its
/// `prompt_eval_count`, `prompt_eval_cached_count` (the part of the prompt
/// read from the daemon's cache) and `eval_count` are the usage.
///
/// A reply with no `thinking` whose `content`, after leading whitespace,
/// starts with `<think>` has that block split out as reasoning, by its tags
/// alone. Content is held only while its start could still become
/// `<think>`; content that cannot is text at once. Inside the block,
/// reasoning streams as it arrives, holding back only trailing whitespace
/// and a partial `</think>`. After `</think>` the rest is text. A block
/// that never closes is reasoning to the end of the reply.
#[derive(Debug, Default)]
pub struct ChatDecoder {
    /// Where the inline split stands.
    split: Split,
    /// Whether the reply called a tool.
    called: bool,
    /// The model the records name.
    model: Option<String>,
}

/// Where a reply's inline reasoning split stands.
#[derive(Debug)]
enum Split {
    /// Content held while its start, after whitespace, could still become
    /// `<think>`.
    Opening(String),
    /// Inside a leading `<think>` block. `held` is trailing whitespace and a
    /// partial `</think>` not yet written; `started` is whether any
    /// reasoning was, so leading whitespace is dropped until then.
    Inside { held: String, started: bool },
    /// Content is text. `trim` drops leading whitespace until visible text
    /// arrives after a split block.
    Text { trim: bool },
}

impl Default for Split {
    fn default() -> Self {
        Self::Opening(String::new())
    }
}

/// The length of the longest proper prefix of `tag` that `text` ends with.
/// The tags are ASCII, so the cut is a character boundary.
fn partial_suffix(text: &str, tag: &str) -> usize {
    (1..tag.len())
        .rev()
        .find(|&len| text.ends_with(&tag[..len]))
        .unwrap_or(0)
}

impl<'id> Decoder<'id, Completion> for ChatDecoder {
    type Event = ChatRecord;

    fn classify(&self, frame: WireFrame) -> WireEvent<ChatRecord> {
        wire::classify_marker_keyed_frame(&frame.as_str(), RECORD_KEYS)
    }

    /// Fields are read leniently: one the record leaves out or sends with
    /// another type is absent, so only `error` and `done` steer the reply.
    fn decode(
        &mut self,
        ChatRecord(fields): ChatRecord,
        mut out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        let record = Value::Object(fields);
        if let Some(error) = record.get("error").filter(|error| !error.is_null()) {
            let body = serde_json::json!({ "error": error }).to_string();
            return Err(ProviderError::from_provider_body(body));
        }
        if let Some(model) = record.str("model").filter(|model| !model.is_empty()) {
            tracing::Span::current().record("gen_ai.response.model", model);
            self.model = Some(model.to_owned());
        }
        let message = record.get("message").unwrap_or(&Value::Null);
        if let Some(thinking) = message
            .str("thinking")
            .filter(|thinking| !thinking.is_empty())
        {
            // A reply with native thinking splits nothing out of its content.
            self.release(&mut out)?;
            reason(thinking, &mut out)?;
        }
        if let Some(content) = message.str("content").filter(|content| !content.is_empty()) {
            self.content(content, &mut out)?;
        }
        for call in message.arr("tool_calls") {
            self.call(call, &mut out)?;
        }
        if record.bool("done") != Some(true) {
            return Ok(Flow::More);
        }
        self.release(&mut out)?;
        out.end_run()?;
        let reason = record.str("done_reason").map(|reason| match reason {
            "stop" if self.called => FinishReason::ToolCalls,
            "stop" => FinishReason::Stop,
            "length" => FinishReason::Length,
            other => FinishReason::Other(other.to_owned()),
        });
        let (input, output) = (record.u64("prompt_eval_count"), record.u64("eval_count"));
        let usage = Usage {
            input_tokens: input,
            output_tokens: output,
            total_tokens: input.zip(output).map(|(input, output)| input + output),
            cached_input_tokens: record.u64("prompt_eval_cached_count"),
            ..Usage::default()
        };
        Ok(out.end(Finish {
            usage,
            reason,
            model: self.model.take(),
            ..Finish::default()
        }))
    }
}

impl ChatDecoder {
    /// Write a fragment of `content`, through the inline split.
    fn content(
        &mut self,
        content: &str,
        out: &mut Out<'_, Completion>,
    ) -> Result<(), ProviderError> {
        match std::mem::replace(&mut self.split, Split::Text { trim: false }) {
            Split::Text { trim } => {
                let text = if trim { content.trim_start() } else { content };
                self.split = Split::Text {
                    trim: trim && text.is_empty(),
                };
                if !text.is_empty() {
                    out.run(Block::Text, text)?;
                }
                Ok(())
            }
            Split::Opening(mut held) => {
                held.push_str(content);
                let trimmed = held.trim_start();
                if let Some(rest) = trimmed.strip_prefix(THINK_OPEN) {
                    let rest = rest.to_owned();
                    self.split = Split::Inside {
                        held: String::new(),
                        started: false,
                    };
                    self.content(&rest, out)
                } else if THINK_OPEN.starts_with(trimmed) {
                    self.split = Split::Opening(held);
                    Ok(())
                } else {
                    out.run(Block::Text, &held)?;
                    Ok(())
                }
            }
            Split::Inside {
                mut held,
                mut started,
            } => {
                held.push_str(content);
                if let Some((reasoning, rest)) = held.split_once(THINK_CLOSE) {
                    write_reasoning(reasoning.trim_end(), started, out)?;
                    self.split = Split::Text { trim: true };
                    return self.content(rest, out);
                }
                let cut = held.len() - partial_suffix(&held, THINK_CLOSE);
                let cut = held[..cut].trim_end().len();
                started |= write_reasoning(&held[..cut], started, out)?;
                self.split = Split::Inside {
                    held: held.split_off(cut),
                    started,
                };
                Ok(())
            }
        }
    }

    /// End the split: content held at the start is text as it arrived, and
    /// what an open block held is reasoning.
    fn release(&mut self, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
        match std::mem::replace(&mut self.split, Split::Text { trim: false }) {
            Split::Opening(held) if !held.is_empty() => {
                out.run(Block::Text, &held)?;
            }
            Split::Inside { held, started } => {
                write_reasoning(held.trim_end(), started, out)?;
            }
            Split::Opening(_) | Split::Text { .. } => {}
        }
        Ok(())
    }

    /// Write one whole call. A call without a name is dropped: nothing can
    /// answer it.
    fn call(&mut self, call: &Value, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
        let Ok(name) = ToolName::new(
            call.at("/function/name")
                .and_then(Value::as_str)
                .unwrap_or_default(),
        ) else {
            tracing::warn!("Ollama sent a tool call without a name; nothing can answer it");
            return Ok(());
        };
        let arguments = match call.at("/function/arguments") {
            None | Some(Value::Null) => "{}".to_owned(),
            Some(Value::String(arguments)) => arguments.clone(),
            Some(arguments) => arguments.to_string(),
        };
        self.release(out)?;
        out.end_run()?;
        self.called = true;
        let id = CallId::from_wire(call.str("id").unwrap_or_default());
        let index = out.fresh_index();
        out.open(index, Block::Call { id, name }, call.clone())?;
        out.push(index, &arguments)?;
        out.finish(index)
    }

    /// Project the verdict and an error envelope off one raw record
    /// before normalization discards them. A payload that is not JSON
    /// projects nothing.
    pub(crate) fn project(payload: &[u8], sink: &mut ObservationSink<'_>) {
        let Ok(record) = serde_json::from_slice::<Value>(payload) else {
            return;
        };
        let verdict = match record.str("done_reason") {
            Some(reason) => AdapterVerdict {
                finish_reason: Some(sink.scrub(reason)),
                block_reason: None,
                detail: None,
                model: record.str("model").map(|model| sink.scrub(model)),
            },
            None => AdapterVerdict::default(),
        };
        sink.provider(verdict, None);
        if let Some(message) = record.str("error") {
            ObservedError {
                code: None,
                kind: None,
                message: Some(message.to_owned()),
            }
            .emit(sink);
        }
    }
}

/// Write a fragment of an inline block's reasoning, its leading whitespace
/// dropped until the block has `started`. Returns whether it wrote any.
fn write_reasoning(
    text: &str,
    started: bool,
    out: &mut Out<'_, Completion>,
) -> Result<bool, ProviderError> {
    let text = if started { text } else { text.trim_start() };
    if text.is_empty() {
        return Ok(false);
    }
    reason(text, out)?;
    Ok(true)
}

/// Write a fragment of reasoning. Its block's item is the `thinking` it
/// joins, so a same-model replay sends it back.
fn reason(text: &str, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
    let index = out.run(Block::Reasoning { redacted: false }, text)?;
    out.edit(index, |item| match item {
        Value::Object(fields) => {
            if let Some(Value::String(thinking)) = fields.get_mut("thinking") {
                thinking.push_str(text);
            }
        }
        _ => *item = serde_json::json!({ "thinking": text }),
    })
}

pub(crate) mod document;

#[cfg(test)]
mod tests;
