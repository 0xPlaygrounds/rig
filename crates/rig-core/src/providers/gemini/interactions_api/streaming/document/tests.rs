//! The Interactions fold on hand-built events. Unit tests: no recording
//! answers one prompt both unary and streamed on this API, so the rules
//! are pinned on events shaped as `corpus_delta/interactions_baseline`
//! records them.

use serde_json::{Value, json};

use super::Interaction;
use crate::wire::WireFrame;
use crate::wire::document::Reassemble;

/// The document `events` add up to.
fn folded(events: impl IntoIterator<Item = Value>) -> Value {
    let mut document = Interaction::default();
    for event in events {
        document.absorb(&WireFrame::Text(event.to_string()));
    }
    document.finish()
}

fn created() -> Value {
    json!({"event_type": "interaction.created", "interaction": {
        "id": "v1_a", "model": "gemini-2.5-flash", "object": "interaction", "status": "in_progress",
    }})
}

fn start(index: u64, step: Value) -> Value {
    json!({"event_type": "step.start", "index": index, "step": step})
}

fn delta(index: u64, delta: Value) -> Value {
    json!({"event_type": "step.delta", "index": index, "delta": delta})
}

fn stop(index: u64) -> Value {
    json!({"event_type": "step.stop", "index": index})
}

fn completed(status: &str) -> Value {
    json!({"event_type": "interaction.completed", "interaction": {
        "created": "1970-01-01T00:00:00Z", "id": "v1_a", "model": "gemini-2.5-flash",
        "object": "interaction", "service_tier": "standard", "status": status,
        "updated": "1970-01-01T00:00:00Z", "usage": {"total_input_tokens": 12, "total_output_tokens": 1},
    }})
}

#[test]
fn a_text_answer_rebuilds_the_unary_interaction() {
    let document = folded([
        created(),
        json!({"event_type": "interaction.status_update", "interaction_id": "v1_a", "status": "in_progress"}),
        start(0, json!({"type": "thought"})),
        delta(
            0,
            json!({"type": "thought_summary", "content": {"type": "text", "text": "**Plan**\n"}}),
        ),
        delta(
            0,
            json!({"type": "thought_summary", "content": {"type": "text", "text": "Say it."}}),
        ),
        delta(0, json!({"type": "thought_signature", "signature": "sig"})),
        stop(0),
        start(1, json!({"type": "model_output"})),
        delta(1, json!({"type": "text", "text": "capt"})),
        delta(1, json!({"type": "text", "text": "ured"})),
        stop(1),
        completed("completed"),
        json!("[DONE]"),
    ]);
    assert_eq!(
        document,
        json!({
            "id": "v1_a", "model": "gemini-2.5-flash", "object": "interaction", "status": "completed",
            "created": "1970-01-01T00:00:00Z", "service_tier": "standard",
            "updated": "1970-01-01T00:00:00Z", "usage": {"total_input_tokens": 12, "total_output_tokens": 1},
            "steps": [
                {"type": "thought", "summary": [{"type": "text", "text": "**Plan**\nSay it."}], "signature": "sig"},
                {"type": "model_output", "content": [{"type": "text", "text": "captured"}]},
            ],
        })
    );
}

#[test]
fn streamed_arguments_become_the_calls_arguments() {
    let document = folded([
        created(),
        start(
            0,
            json!({"type": "function_call", "id": "c1", "name": "add", "arguments": {}}),
        ),
        delta(
            0,
            json!({"type": "arguments_delta", "arguments": "{\"y\":25,"}),
        ),
        delta(
            0,
            json!({"type": "arguments_delta", "arguments": "\"x\":17}"}),
        ),
        stop(0),
        completed("requires_action"),
    ]);
    assert_eq!(
        document["steps"],
        json!([{"type": "function_call", "id": "c1", "name": "add", "arguments": {"y": 25, "x": 17}}])
    );
    assert_eq!(document["status"], "requires_action");
}

#[test]
fn a_delta_without_a_start_opens_the_step_it_implies() {
    let document = folded([
        delta(0, json!({"type": "thought_signature", "signature": "sig"})),
        delta(1, json!({"type": "text", "text": "42"})),
        delta(
            1,
            json!({"type": "image", "mime_type": "image/png", "data": "AA=="}),
        ),
    ]);
    assert_eq!(
        document["steps"],
        json!([
            {"type": "thought", "signature": "sig"},
            {"type": "model_output", "content": [
                {"type": "text", "text": "42"},
                {"type": "image", "mime_type": "image/png", "data": "AA=="},
            ]},
        ])
    );
}

#[test]
fn steps_the_completed_interaction_states_win() {
    let mut last = completed("completed");
    last["interaction"]["steps"] =
        json!([{"type": "model_output", "content": [{"type": "text", "text": "whole"}]}]);
    let document = folded([
        start(0, json!({"type": "model_output"})),
        delta(0, json!({"type": "text", "text": "part"})),
        last.clone(),
    ]);
    assert_eq!(document["steps"], last["interaction"]["steps"]);
}

#[test]
fn a_whole_interaction_is_the_document_as_sent() {
    let whole = json!({"id": "v1_a", "status": "completed", "steps": []});
    assert_eq!(folded([whole.clone()]), whole);
}

#[test]
fn a_cut_stream_keeps_what_arrived_and_an_empty_one_has_no_document() {
    let document = folded([
        created(),
        start(0, json!({"type": "model_output"})),
        delta(0, json!({"type": "text", "text": "par"})),
    ]);
    assert_eq!(document["status"], "in_progress");
    assert_eq!(document["steps"][0]["content"][0]["text"], "par");
    assert_eq!(folded([]), Value::Null);
}
