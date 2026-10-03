//! The Gemini Interactions wire's history suite. A unary reply is the whole
//! interaction resource; a streamed one is its `step.*` events, ended by
//! `interaction.completed`.

use rig_core::completion::CompletionRequest;
use rig_core::error::EncodeError;
use rig_core::message::AssistantContent;
use rig_core::providers::gemini::GeminiConfig;
use rig_core::providers::gemini::interactions_api::Interactions;
use rig_core::wire::{Mode, Wire, WireFrame};
use rig_history_conformance::{
    Ablation, CallShape, Ending, HistoryFixture, Rng, Shape, http_body, replies,
};
use serde_json::{Value, json};

pub struct InteractionsHistory;

const MODEL: &str = "gemini-3-flash-preview";

fn frame(value: Value) -> WireFrame {
    WireFrame::Text(value.to_string())
}

fn thought(text: &str, signature: &str) -> Value {
    json!({"type": "thought", "signature": signature, "summary": [{"type": "text", "text": text}]})
}

fn output(text: &str) -> Value {
    json!({"type": "model_output", "content": [{"type": "text", "text": text}]})
}

fn call(id: &str, arguments: Value) -> Value {
    json!({"type": "function_call", "id": id, "name": "lookup", "arguments": arguments})
}

/// A model output that holds text and an image, one block each.
fn mixed_output() -> Value {
    json!({"type": "model_output", "content": [
        {"type": "text", "text": "Here is the chart."},
        {"type": "image", "data": "aW1n", "mime_type": "image/png"},
    ]})
}

fn search_call() -> Value {
    json!({"type": "google_search_call", "id": "gs_1", "arguments": {"queries": ["rig"]}})
}

fn search_result() -> Value {
    json!({"type": "google_search_result", "call_id": "gs_1", "signature": "c2ln",
        "result": [{"url": "https://example.com", "title": "Rig"}]})
}

fn usage() -> Value {
    json!({"total_input_tokens": 10, "total_output_tokens": 5, "total_thought_tokens": 2,
        "total_tokens": 17})
}

/// The whole interaction resource holding `steps`, ended with `status`.
fn resource(steps: Vec<Value>, status: &str) -> Value {
    json!({"id": "int_1", "model": MODEL, "object": "interaction", "status": status,
        "steps": steps, "usage": usage()})
}

fn steps_of(shape: Shape) -> (Vec<Value>, &'static str) {
    match shape {
        Shape::Rich => (
            vec![
                thought("plan the lookup", "c2lnX3JpY2g="),
                mixed_output(),
                search_call(),
                search_result(),
                call("call_1", json!({"q": "rig"})),
            ],
            "requires_action",
        ),
        Shape::Interleaved => (
            vec![
                thought("first", "c2lnXzE="),
                output("between"),
                thought("second", "c2lnXzI="),
                call("call_1", json!({"q": "rig"})),
            ],
            "requires_action",
        ),
        Shape::Unknown => (
            vec![
                json!({"type": "x_rig_invented", "id": "x_1", "frames": [1, 2]}),
                json!({"type": "model_output", "x_rig_field": true,
                    "content": [{"type": "text", "text": "noted"}]}),
            ],
            "completed",
        ),
    }
}

/// `step` restated as the events a stream carries for it: a start naming
/// its type, its content as deltas, and a stop.
fn step_events(index: usize, step: &Value) -> Vec<Value> {
    let start = |step: Value| json!({"event_type": "step.start", "index": index, "step": step});
    let delta = |delta: Value| json!({"event_type": "step.delta", "index": index, "delta": delta});
    let stop = json!({"event_type": "step.stop", "index": index});
    let kind = step["type"].as_str().unwrap_or_default();
    let mut events = match kind {
        "thought" => {
            let mut events = vec![start(json!({"type": "thought"}))];
            for summary in step["summary"].as_array().into_iter().flatten() {
                events.push(delta(
                    json!({"type": "thought_summary", "content": summary}),
                ));
            }
            events.push(delta(
                json!({"type": "thought_signature", "signature": step["signature"]}),
            ));
            events
        }
        "model_output" => {
            let mut opening = step.clone();
            if let Some(fields) = opening.as_object_mut() {
                fields.shift_remove("content");
            }
            let mut events = vec![start(opening)];
            for item in step["content"].as_array().into_iter().flatten() {
                match item["text"].as_str() {
                    Some(text) => {
                        let (head, tail) = text.split_at(text.len() / 2);
                        events.push(delta(json!({"type": "text", "text": head})));
                        events.push(delta(json!({"type": "text", "text": tail})));
                    }
                    None => events.push(delta(item.clone())),
                }
            }
            events
        }
        "function_call" => {
            let mut opening = step.clone();
            opening["arguments"] = json!({});
            vec![
                start(opening),
                delta(
                    json!({"type": "arguments_delta", "arguments": step["arguments"].to_string()}),
                ),
            ]
        }
        _ => vec![start(step.clone())],
    };
    events.push(stop);
    events
}

/// The stream of `steps`, opened by `interaction.created` and ended by
/// `interaction.completed` with `status`.
fn stream(steps: &[Value], status: &str) -> Vec<WireFrame> {
    let mut events = vec![json!({"event_type": "interaction.created",
        "interaction": {"id": "int_1", "model": MODEL, "object": "interaction",
            "status": "in_progress"}})];
    for (index, step) in steps.iter().enumerate() {
        events.extend(step_events(index, step));
    }
    events.push(json!({"event_type": "interaction.completed",
        "interaction": {"id": "int_1", "model": MODEL, "object": "interaction",
            "status": status, "usage": usage()}}));
    events.into_iter().map(frame).collect()
}

impl HistoryFixture for InteractionsHistory {
    type Wire = Interactions;

    fn wire(&self, model: &str) -> Interactions {
        Interactions::new(GeminiConfig::new("test-key"), model)
    }

    fn reply_spec(&self, rng: &mut Rng) -> Option<replies::Spec> {
        Some(replies::interactions_spec(rng))
    }

    fn reply_frames(&self, spec: &replies::Spec) -> Option<replies::Frames<WireFrame>> {
        let (whole, streamed) = replies::interactions_build(self.model(), spec);
        Some(replies::Frames {
            whole: replies::values(whole),
            streamed: replies::values(streamed),
        })
    }

    fn model(&self) -> &'static str {
        MODEL
    }

    fn other_model(&self) -> &'static str {
        "gemini-2.5-flash"
    }

    fn body(
        &self,
        wire: &Interactions,
        request: CompletionRequest,
        mode: Mode,
    ) -> Result<Value, EncodeError> {
        http_body(&wire.encode(request, mode)?)
    }

    fn reply(&self, shape: Shape, mode: Mode) -> Option<Vec<WireFrame>> {
        let (steps, status) = steps_of(shape);
        Some(match mode {
            Mode::Unary => vec![frame(resource(steps, status))],
            Mode::Streaming => stream(&steps, status),
        })
    }

    /// A unary reply states its arguments as JSON, so only argument text
    /// that is JSON has a whole form; a stream carries any text.
    fn call_reply(&self, arguments: &str, mode: Mode) -> Option<Vec<WireFrame>> {
        match mode {
            Mode::Unary => {
                let arguments: Value = serde_json::from_str(arguments).ok()?;
                Some(vec![frame(resource(
                    vec![call("call_1", arguments)],
                    "requires_action",
                ))])
            }
            Mode::Streaming => {
                let events = [
                    json!({"event_type": "step.start", "index": 0,
                        "step": {"type": "function_call", "id": "call_1", "name": "lookup"}}),
                    json!({"event_type": "step.delta", "index": 0,
                        "delta": {"type": "arguments_delta", "arguments": arguments}}),
                    json!({"event_type": "step.stop", "index": 0}),
                    json!({"event_type": "interaction.completed",
                        "interaction": {"id": "int_1", "model": MODEL,
                            "status": "requires_action"}}),
                ];
                Some(events.into_iter().map(frame).collect())
            }
        }
    }

    /// Every documented `status`, the deprecated `budget_exceeded`, an
    /// unknown one, and none.
    fn finishes(&self) -> Vec<(&'static str, Vec<WireFrame>, Ending)> {
        let ended = |status: &str| {
            let mut document = resource(vec![output("done")], status);
            if status.is_empty()
                && let Some(fields) = document.as_object_mut()
            {
                fields.shift_remove("status");
            }
            vec![frame(document)]
        };
        let called = vec![frame(resource(
            vec![call("call_1", json!({}))],
            "requires_action",
        ))];
        vec![
            ("completed", ended("completed"), Ending::Success),
            ("requires_action", called, Ending::Success),
            ("incomplete", ended("incomplete"), Ending::Success),
            ("budget_exceeded", ended("budget_exceeded"), Ending::Success),
            ("failed", ended("failed"), Ending::Failure),
            ("cancelled", ended("cancelled"), Ending::Failure),
            ("in_progress", ended("in_progress"), Ending::Failure),
            ("queued", ended("queued"), Ending::Failure),
            ("x_rig_status", ended("x_rig_status"), Ending::Failure),
            ("no status", ended(""), Ending::Failure),
        ]
    }

    /// No field of the resource is required: a step without its type is
    /// an opaque step, a call without a name is dropped, and a resource
    /// without a status is a failed turn.
    fn ablation(&self) -> Option<Ablation<WireFrame>> {
        let (steps, status) = steps_of(Shape::Rich);
        Some(Ablation {
            document: resource(steps, status),
            required: &[],
            frames: |document| vec![frame(document)],
        })
    }

    /// A step is its own whole interaction.
    fn decode_item(&self, block: &AssistantContent) -> Option<AssistantContent> {
        let step = block.native_item()?.clone();
        let response = rig_core::test_utils::history::decode(
            &self.wire(MODEL),
            Mode::Unary,
            [frame(resource(vec![step], "completed"))],
        )
        .ok()?;
        response.choice.into_iter().next()
    }

    /// Two calls to `weather`: listed in one resource, or streamed as two
    /// steps, at their own indices or both at index 0.
    fn calls_reply(&self, shape: CallShape, mode: Mode) -> Option<Vec<WireFrame>> {
        let weather = |id: &str, city: &str| {
            json!({"type": "function_call", "id": id, "name": "weather",
                "arguments": {"city": city}})
        };
        let calls = vec![weather("a1", "Paris"), weather("b2", "Rome")];
        match (shape, mode) {
            (CallShape::WholeList, Mode::Unary) => {
                Some(vec![frame(resource(calls, "requires_action"))])
            }
            (CallShape::WholeList, Mode::Streaming) => Some(stream(&calls, "requires_action")),
            (CallShape::ReusedIndex, Mode::Streaming) => {
                let mut events: Vec<Value> =
                    calls.iter().flat_map(|call| step_events(0, call)).collect();
                events.push(json!({"event_type": "interaction.completed",
                    "interaction": {"id": "int_1", "model": MODEL, "status": "requires_action"}}));
                Some(events.into_iter().map(frame).collect())
            }
            _ => None,
        }
    }

    fn empty_reply(&self, mode: Mode) -> Option<Vec<WireFrame>> {
        Some(match mode {
            Mode::Unary => vec![frame(resource(Vec::new(), "completed"))],
            Mode::Streaming => stream(&[], "completed"),
        })
    }

    fn finish_reason_pointer(&self) -> Option<&'static str> {
        Some("/status")
    }

    fn error_frame(&self) -> Option<WireFrame> {
        Some(frame(json!({"event_type": "error",
            "error": {"code": "unavailable", "message": "overloaded"}})))
    }
}

rig_history_conformance::history_conformance_suite! {
    wire: "gemini_interactions",
    fixture: InteractionsHistory,
}
