//! The Chat Completions reassembler.
//!
//! Interim: [`TerminalRecord`] rebuilds the streamed `raw` this wire wrote
//! before replies were reassembled, so no recorded value moves with the
//! mechanism. The Chat family replaces it with the `chat.completion`
//! reassembler of TYPED_OPTIONS.md section 9.

use serde_json::{Map, Value};

use super::{ChatDecoder, ChatEvent, Quirks};
use crate::completion::FinishReason;
use crate::json_utils::Lenient;
use crate::wire::document::Reassemble;
use crate::wire::{Decoder, WireEvent, WireFrame};

/// Interim Chat reassembler: the terminal record of a chunked reply
/// (`usage`, rig's `finish_reason`, `response_id`, `model`, `logprobs` and
/// the other top-level chunk fields as `additional_params`). A whole or
/// bare-string reply, and a reply that failed in band or did not end,
/// rebuild nothing.
#[derive(Default)]
pub struct TerminalRecord {
    /// The decoder's own metadata fold and classifier, fed the same frames.
    metadata: ChatDecoder,
    /// Whether the primary choice streamed a call.
    called: bool,
    /// Whether the `[DONE]` sentinel arrived.
    done: bool,
    /// Whether a whole or bare-string frame answered, or a frame failed.
    nothing: bool,
}

impl TerminalRecord {
    /// A reassembler for a dialect with `quirks`.
    pub(super) fn new(quirks: Quirks) -> Self {
        Self {
            metadata: ChatDecoder::new(quirks),
            ..Self::default()
        }
    }
}

impl Reassemble<WireFrame> for TerminalRecord {
    fn absorb(&mut self, frame: &WireFrame) {
        match self.metadata.classify(frame.clone()) {
            WireEvent::Known(ChatEvent::Chunk(chunk)) => {
                self.metadata.chunked = true;
                let calls = self.metadata.absorb(&chunk).and_then(|choice| {
                    choice
                        .obj("delta")
                        .and_then(|delta| delta.get("tool_calls"))
                        .and_then(Value::as_array)
                        .map(|calls| !calls.is_empty())
                });
                self.called |= calls.unwrap_or(false);
            }
            WireEvent::Known(ChatEvent::Done) => self.done = true,
            WireEvent::Known(
                ChatEvent::Whole(_) | ChatEvent::BareText(_) | ChatEvent::Failure(_),
            ) => {
                self.nothing = true;
            }
            // The driver reports what does not classify.
            _ => {}
        }
    }

    fn finish(self) -> Value {
        let Self {
            metadata: mut fold,
            called,
            done,
            nothing,
        } = self;
        if nothing {
            return Value::Null;
        }
        // The decoder's `[DONE]` rule for a dialect whose streams omit the
        // finish reason.
        if done && !fold.ended && fold.chunked && fold.quirks.done_without_finish_reason {
            fold.finish = Some(if called {
                FinishReason::ToolCalls
            } else {
                FinishReason::Stop
            });
            fold.ended = true;
        }
        if !fold.ended {
            return Value::Null;
        }
        let fields = std::mem::take(&mut fold.fields);
        let record = [
            ("usage", fold.usage.take()),
            (
                "finish_reason",
                fold.finish
                    .as_ref()
                    .and_then(|reason| serde_json::to_value(reason).ok()),
            ),
            ("response_id", fold.response_id.take().map(Value::String)),
            ("model", fold.response_model.take().map(Value::String)),
            ("logprobs", fold.logprobs.take().map(Value::Object)),
            (
                "additional_params",
                (!fields.is_empty()).then_some(Value::Object(fields)),
            ),
        ];
        Value::Object(
            record
                .into_iter()
                .filter_map(|(key, value)| Some((key.to_owned(), value?)))
                .collect::<Map<String, Value>>(),
        )
    }
}
