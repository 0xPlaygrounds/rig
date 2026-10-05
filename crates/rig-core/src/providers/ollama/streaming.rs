//! The decoder of native Ollama chat replies: a whole `/api/chat` body, or a
//! stream of NDJSON records of the same shape, the last one `done`.
//!
//! A record's `thinking` is reasoning, its `content` text and its
//! `tool_calls` whole calls. A reply with no `thinking` may carry its
//! reasoning as a leading `<think>…</think>` block in `content`; that block
//! is split out as reasoning (see [`ChatDecoder`]).
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
use crate::wire::{
    AdapterEvent, AdapterUsage, AdapterVerdict, Decoder, Flow, ObservationSink, Out, WireEvent,
    WireFrame,
};

/// The keys that make a JSON line a chat record: every record carries
/// `message` or `done`, and an in-band failure carries `error`.
const RECORD_KEYS: &[&str] = &["message", "done", "error"];

/// The marker that opens an inline reasoning block.
const THINK_OPEN: &str = "<think>";
/// The marker that closes it.
const THINK_CLOSE: &str = "</think>";
/// The exact boundary after a `qwen3` reasoning block whose opening marker
/// the chat template prefilled.
const QWEN3_BOUNDARY: &str = "\n</think>\n\n";

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
/// A reply with no `thinking` has its leading, terminated
/// `<think>…</think>` block split out of `content` as reasoning; for a
/// `qwen3` model, whose chat template prefills the opening marker, so is
/// text that ends at the exact boundary `\n</think>\n\n`. Content is held
/// while it could still be such a block: once the block closes it is
/// reasoning and the rest streams as text, and content that cannot be one,
/// or a reply that ends before the block closes, is text as it arrived.
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
    /// Content held while it could still be a leading reasoning block.
    Holding(String),
    /// Content is text. `trim` drops leading whitespace until visible text
    /// arrives after a split block.
    Text { trim: bool },
}

impl Default for Split {
    fn default() -> Self {
        Self::Holding(String::new())
    }
}

/// What the content held so far is.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Held<'a> {
    /// It may still be a leading reasoning block.
    Open,
    /// A leading reasoning block, then visible text, each trimmed.
    Split(&'a str, &'a str),
    /// It cannot be one: it is text as it stands.
    Text,
}

/// Read `content`, held from the start of a reply, as a leading reasoning
/// block. `qwen3` also accepts a block whose opening marker the template
/// prefilled, closed by the exact [`QWEN3_BOUNDARY`]. An unterminated
/// `<think>` stays open, so a whole reply keeps it as text.
pub(crate) fn leading_reasoning(content: &str, qwen3: bool) -> Held<'_> {
    let trimmed = content.trim_start();
    let split = if let Some(rest) = trimmed.strip_prefix(THINK_OPEN) {
        match rest.split_once(THINK_CLOSE) {
            Some(split) => split,
            None => return Held::Open,
        }
    } else if THINK_OPEN.starts_with(trimmed) {
        return Held::Open;
    } else if qwen3 {
        match trimmed.split_once(QWEN3_BOUNDARY) {
            Some(split) => split,
            None => return Held::Open,
        }
    } else {
        return Held::Text;
    };
    Held::Split(split.0.trim(), split.1.trim_start())
}

/// Whether `model` names a `qwen3` model.
fn is_qwen3(model: &str) -> bool {
    model.to_ascii_lowercase().contains("qwen3")
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
        out.raw(record);
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
        let held = match &mut self.split {
            Split::Text { trim } => {
                let text = if *trim { content.trim_start() } else { content };
                if !text.is_empty() {
                    *trim = false;
                    out.run(Block::Text, text)?;
                }
                return Ok(());
            }
            Split::Holding(held) => {
                held.push_str(content);
                std::mem::take(held)
            }
        };
        let qwen3 = self.model.as_deref().is_some_and(is_qwen3);
        match leading_reasoning(&held, qwen3) {
            Held::Open => self.split = Split::Holding(held),
            Held::Text => {
                self.split = Split::Text { trim: false };
                out.run(Block::Text, &held)?;
            }
            Held::Split(reasoning, visible) => {
                if !reasoning.is_empty() {
                    reason(reasoning, out)?;
                }
                self.split = Split::Text {
                    trim: visible.is_empty(),
                };
                if !visible.is_empty() {
                    out.run(Block::Text, visible)?;
                }
            }
        }
        Ok(())
    }

    /// End the split: content still held is text as it arrived.
    fn release(&mut self, out: &mut Out<'_, Completion>) -> Result<(), ProviderError> {
        if let Split::Holding(held) =
            std::mem::replace(&mut self.split, Split::Text { trim: false })
            && !held.is_empty()
        {
            out.run(Block::Text, &held)?;
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

    /// Project usage, the verdict and an error envelope off one raw record
    /// before normalization discards them. A payload that is not JSON
    /// projects nothing.
    pub(crate) fn project(payload: &[u8], sink: &mut ObservationSink<'_>) {
        let Ok(record) = serde_json::from_slice::<Value>(payload) else {
            return;
        };
        let input = record.u64("prompt_eval_count");
        let output = record.u64("eval_count");
        if input.is_some() || output.is_some() {
            sink.emit(AdapterEvent::Usage {
                usage: AdapterUsage {
                    input_tokens: input,
                    output_tokens: output,
                    total_tokens: input.zip(output).map(|(input, output)| input + output),
                    cached_input_tokens: record.u64("prompt_eval_cached_count"),
                    reasoning_tokens: None,
                    tool_input_tokens: None,
                },
            });
        }
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

#[cfg(test)]
mod tests;
