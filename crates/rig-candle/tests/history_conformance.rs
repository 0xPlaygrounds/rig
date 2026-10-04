//! The history conformance suite for candle's local generation wire, on the
//! Qwen3 protocol, which renders reasoning and tools. A local model keeps no
//! provider items, so the rows that check verbatim replay check that nothing
//! breaks. The body is the prompt the model reads, beside the model the
//! request addresses.

#![allow(clippy::expect_used)]

use rig_candle::{
    CandleCompletionResponse, CandleFrame, ConversationProtocol,
    FinishReason as CandleFinishReason, Generation, GenerationEvent,
};
use rig_core::completion::CompletionRequest;
use rig_core::error::EncodeError;
use rig_core::message::{CallId, Reasoning, ToolCall, ToolFunction, ToolName};
use rig_core::wire::{Mode, Wire};
use rig_history_conformance::{Ending, HistoryFixture, Rng, Shape, replies};
use serde_json::Value;

struct CandleHistory;

fn finish(finish_reason: CandleFinishReason) -> CandleFrame {
    CandleFrame::Event(GenerationEvent::Final(CandleCompletionResponse {
        text: String::new(),
        prompt_tokens: 10,
        generated_tokens: 5,
        requested_max_tokens: 256,
        effective_max_tokens: 256,
        finish_reason,
        prefill_duration_ms: 1,
        time_to_first_token_ms: Some(1),
        generation_duration_ms: 2,
        tokens_per_second: None,
    }))
}

fn call(arguments: &str) -> GenerationEvent {
    GenerationEvent::ToolCall(ToolCall::new(
        CallId::from_wire("call_1"),
        ToolFunction::parse(ToolName::new("lookup").expect("a tool name"), arguments),
    ))
}

impl HistoryFixture for CandleHistory {
    type Wire = Generation;

    fn wire(&self, model: &str) -> Generation {
        Generation {
            model: model.to_owned(),
            protocol: ConversationProtocol::Qwen3,
        }
    }

    /// Text events split in the stream; reasoning events arrive whole, as
    /// Candle's contract states.
    fn reply_spec(&self, rng: &mut Rng) -> Option<replies::Spec> {
        let mut blocks = Vec::new();
        for _ in 0..rng.range(0, 4) {
            let kind = rng.pick(&["reasoning", "text", "call"]);
            blocks.push(serde_json::json!({
                "kind": kind,
                "text": replies::maybe_blank(rng),
                "args": replies::args(rng),
            }));
        }
        let finish = rng.pick(&["eos", "max_tokens"]).to_owned();
        Some(replies::Spec {
            blocks,
            finish,
            seed: 0,
        })
    }

    fn reply_frames(&self, spec: &replies::Spec) -> Option<replies::Frames<CandleFrame>> {
        Some(replies::Frames {
            whole: candle_events(spec, false),
            streamed: candle_events(spec, true),
        })
    }

    fn model(&self) -> &'static str {
        "qwen3-00000000000000a1"
    }

    fn other_model(&self) -> &'static str {
        "qwen3-00000000000000b2"
    }

    /// The payload is the request itself, rendered by the runtime; the body
    /// is that rendered prompt and the model the request addresses, which
    /// the runtime refuses when it is not the loaded one.
    fn body(
        &self,
        wire: &Generation,
        request: CompletionRequest,
        mode: Mode,
    ) -> Result<Value, EncodeError> {
        let request = wire.encode(request, mode)?;
        let prompt = wire
            .prompt(&request)
            .map_err(|error| EncodeError::request(error.to_string()))?;
        Ok(serde_json::json!({"model": request.model, "prompt": prompt}))
    }

    /// The generator restates nothing: a whole reply is its events, read at
    /// once.
    fn reply(&self, shape: Shape, _mode: Mode) -> Option<Vec<CandleFrame>> {
        let events = match shape {
            Shape::Rich => vec![
                GenerationEvent::Reasoning(Reasoning::new("plan the lookup")),
                GenerationEvent::Text("looking it up".to_owned()),
                call(r#"{"q": "rig"}"#),
            ],
            Shape::Interleaved => vec![
                GenerationEvent::Reasoning(Reasoning::new("first")),
                GenerationEvent::Text("between".to_owned()),
                GenerationEvent::Reasoning(Reasoning::new("second")),
                call(r#"{"q": "rig"}"#),
            ],
            Shape::Unknown => return None,
        };
        Some(
            events
                .into_iter()
                .map(CandleFrame::Event)
                .chain([finish(CandleFinishReason::Eos)])
                .collect(),
        )
    }

    fn call_reply(&self, arguments: &str, _mode: Mode) -> Option<Vec<CandleFrame>> {
        Some(vec![
            CandleFrame::Event(call(arguments)),
            finish(CandleFinishReason::Eos),
        ])
    }

    fn finishes(&self) -> Vec<(&'static str, Vec<CandleFrame>, Ending)> {
        vec![
            (
                "eos",
                vec![finish(CandleFinishReason::Eos)],
                Ending::Success,
            ),
            (
                "max_tokens",
                vec![finish(CandleFinishReason::MaxTokens)],
                Ending::Success,
            ),
        ]
    }

    fn keeps_natives(&self) -> bool {
        false
    }

    fn no_tool_calls(&self) -> Option<&'static str> {
        Some("the body is a rendered prompt string, which holds calls and results as text")
    }
}

rig_history_conformance::history_conformance_suite! {
    wire: "candle",
    fixture: CandleHistory,
}

/// `spec`'s events, text split into pieces when `split`.
fn candle_events(spec: &replies::Spec, split: bool) -> Vec<CandleFrame> {
    let mut rng = Rng::new(spec.seed);
    let mut events = Vec::new();
    for (k, block) in spec.blocks.iter().enumerate() {
        let text = block["text"].as_str().unwrap_or_default().to_owned();
        match block["kind"].as_str() {
            Some("reasoning") => events.push(GenerationEvent::Reasoning(Reasoning::new(&text))),
            Some("text") => {
                let pieces = if split { rng.split(&text) } else { vec![text] };
                events.extend(pieces.into_iter().map(GenerationEvent::Text));
            }
            _ => events.push(GenerationEvent::ToolCall(ToolCall::new(
                CallId::from_wire(format!("call_{k}")),
                ToolFunction::parse(
                    ToolName::new("lookup").expect("a tool name"),
                    &block["args"].to_string(),
                ),
            ))),
        }
    }
    let reason = if spec.finish == "eos" {
        CandleFinishReason::Eos
    } else {
        CandleFinishReason::MaxTokens
    };
    events
        .into_iter()
        .map(CandleFrame::Event)
        .chain([finish(reason)])
        .collect()
}
