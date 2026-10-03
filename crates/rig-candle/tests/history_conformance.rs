//! The history conformance suite for candle's local generation wire. A local
//! model keeps no provider items, so the rows that check verbatim replay
//! check that nothing breaks; every other invariant holds as on any wire.

#![allow(clippy::expect_used)]

use rig_candle::{
    CandleCompletionResponse, CandleFrame, FinishReason as CandleFinishReason, Generation,
    GenerationEvent,
};
use rig_core::completion::CompletionRequest;
use rig_core::error::EncodeError;
use rig_core::message::{CallId, Reasoning, ToolCall, ToolFunction, ToolName};
use rig_core::wire::{Mode, Wire};
use rig_history_conformance::{Ending, HistoryFixture, Shape};
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

    fn wire(&self, _model: &str) -> Generation {
        Generation
    }

    fn model(&self) -> &'static str {
        "qwen3-local"
    }

    fn other_model(&self) -> &'static str {
        "llama-local"
    }

    fn body(
        &self,
        wire: &Generation,
        request: CompletionRequest,
        mode: Mode,
    ) -> Result<Value, EncodeError> {
        Ok(serde_json::to_value(wire.encode(request, mode)?)?)
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
}

rig_history_conformance::history_conformance_suite! {
    wire: "candle",
    fixture: CandleHistory,
}
