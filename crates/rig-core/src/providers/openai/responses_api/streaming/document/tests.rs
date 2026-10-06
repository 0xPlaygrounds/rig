//! The Responses reassembler over hand-built event streams: what each
//! lifecycle and item event adds to the rebuilt `Response`, and parity on
//! the route no recorded pair covers (ChatGPT, which answers both ways with
//! an event stream).

use serde_json::{Value, json};

use super::Response;
use crate::providers::chatgpt::DIALECT as CHATGPT;
use crate::providers::openai::wire::OpenAIConfig;
use crate::test_utils::raw_parity::raw_pair;
use crate::wire::WireFrame;
use crate::wire::document::Reassemble;

/// The document `events` rebuild, each sent as one frame.
fn rebuilt(events: &[Value]) -> Value {
    let mut document = Response::default();
    for event in events {
        document.absorb(&WireFrame::Text(event.to_string()));
    }
    document.finish()
}

/// `events` as the event stream a provider sends.
fn sse(events: &[Value]) -> String {
    events
        .iter()
        .map(|event| {
            let kind = event["type"].as_str().unwrap_or_default();
            format!("event: {kind}\ndata: {event}\n\n")
        })
        .collect()
}

fn lifecycle(kind: &str, response: Value) -> Value {
    json!({"type": kind, "sequence_number": 0, "response": response})
}

fn item(kind: &str, index: u64, item: Value) -> Value {
    json!({"type": kind, "sequence_number": 1, "output_index": index, "item": item})
}

fn message(id: &str, text: &str) -> Value {
    json!({
        "type": "message",
        "id": id,
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": [], "logprobs": []}]
    })
}

fn call(id: &str, arguments: &str) -> Value {
    json!({
        "type": "function_call",
        "id": id,
        "call_id": format!("call_{id}"),
        "name": "lookup",
        "arguments": arguments,
        "status": "completed"
    })
}

fn envelope(status: &str, output: Value) -> Value {
    json!({
        "id": "resp_1",
        "object": "response",
        "model": "gpt-5.4",
        "status": status,
        "output": output,
        "usage": {"input_tokens": 9, "output_tokens": 2, "total_tokens": 11},
        "service_tier": "default"
    })
}

#[test]
fn a_completed_stream_is_its_terminal_response() {
    let terminal = envelope("completed", json!([message("msg_1", "pong")]));
    let document = rebuilt(&[
        lifecycle("response.created", envelope("in_progress", json!([]))),
        item(
            "response.output_item.added",
            0,
            json!({"type": "message", "id": "msg_1", "content": []}),
        ),
        json!({"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": "pong", "obfuscation": "x"}),
        item("response.output_item.done", 0, message("msg_1", "pong")),
        lifecycle("response.completed", terminal.clone()),
    ]);
    assert_eq!(document, terminal);
}

/// The ChatGPT backend's terminal states an empty `output`: the items the
/// stream finished take their places by output index, whatever order they
/// finished in.
#[test]
fn an_empty_terminal_output_is_the_done_items_by_index() {
    let document = rebuilt(&[
        lifecycle("response.created", envelope("in_progress", json!([]))),
        item(
            "response.output_item.added",
            0,
            json!({"type": "reasoning", "id": "rs_1", "summary": []}),
        ),
        item(
            "response.output_item.added",
            1,
            json!({"type": "function_call", "id": "fc_1", "arguments": ""}),
        ),
        item("response.output_item.done", 1, call("fc_1", "{\"q\":1}")),
        item(
            "response.output_item.done",
            0,
            json!({"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "gAAA"}),
        ),
        lifecycle("response.completed", envelope("completed", json!([]))),
    ]);
    assert_eq!(
        document,
        envelope(
            "completed",
            json!([
                {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "gAAA"},
                call("fc_1", "{\"q\":1}")
            ])
        )
    );
}

#[test]
fn a_terminal_that_states_its_output_keeps_it() {
    let terminal = envelope("completed", json!([message("msg_1", "final")]));
    let document = rebuilt(&[
        item("response.output_item.done", 0, message("msg_1", "streamed")),
        lifecycle("response.completed", terminal.clone()),
    ]);
    assert_eq!(document, terminal);
}

#[test]
fn an_incomplete_terminal_is_incomplete_whatever_its_status_says() {
    let mut terminal = envelope("completed", json!([message("msg_1", "po")]));
    terminal["incomplete_details"] = json!({"reason": "max_output_tokens"});
    let document = rebuilt(&[lifecycle("response.incomplete", terminal.clone())]);
    terminal["status"] = json!("incomplete");
    assert_eq!(document, terminal);
}

#[test]
fn a_failed_stream_is_its_failed_response() {
    let mut failed = envelope("failed", json!([]));
    failed["error"] = json!({"code": "server_error", "message": "boom"});
    let document = rebuilt(&[
        lifecycle("response.created", envelope("in_progress", json!([]))),
        lifecycle("response.failed", failed.clone()),
    ]);
    assert_eq!(document, failed);
}

/// A stream cut before its terminal records the response so far, with the
/// items as last stated; a done item replaces the item it was added as.
#[test]
fn a_cut_stream_is_the_response_so_far() {
    let added = json!({"type": "message", "id": "msg_2", "content": []});
    let document = rebuilt(&[
        lifecycle("response.created", envelope("queued", json!([]))),
        lifecycle("response.in_progress", envelope("in_progress", json!([]))),
        item(
            "response.output_item.added",
            0,
            json!({"type": "message", "id": "msg_1", "content": []}),
        ),
        item("response.output_item.done", 0, message("msg_1", "pong")),
        item(
            "response.output_item.added",
            0,
            json!({"type": "message", "id": "late"}),
        ),
        item("response.output_item.added", 1, added.clone()),
    ]);
    assert_eq!(
        document,
        envelope("in_progress", json!([message("msg_1", "pong"), added]))
    );
}

/// A whole response body (a WebSocket `response.done`, or a unary body) is
/// the document, and nothing after a terminal replaces it.
#[test]
fn a_whole_body_is_the_document_and_a_terminal_is_final() {
    let body = envelope("completed", json!([message("msg_1", "pong")]));
    let document = rebuilt(&[
        body.clone(),
        lifecycle("response.in_progress", envelope("in_progress", json!([]))),
    ]);
    assert_eq!(document, body);

    let terminal = envelope("completed", json!([message("msg_1", "pong")]));
    let document = rebuilt(&[
        lifecycle("response.completed", terminal.clone()),
        lifecycle("response.created", envelope("in_progress", json!([]))),
    ]);
    assert_eq!(document, terminal);
}

#[test]
fn a_stream_with_no_response_records_nothing() {
    assert_eq!(rebuilt(&[]), Value::Null);
    let error = json!({"type": "error", "code": "rate_limit_exceeded", "message": "slow down"});
    assert_eq!(rebuilt(&[error]), Value::Null);
    let text = json!({"type": "response.output_text.delta", "output_index": 0, "delta": "x"});
    assert_eq!(rebuilt(&[text]), Value::Null);
}

/// Items with no response around them still rebuild a response.
#[test]
fn items_without_a_lifecycle_event_rebuild_a_bare_response() {
    let document = rebuilt(&[json!({
        "type": "response.output_item.done",
        "item": message("msg_1", "pong")
    })]);
    assert_eq!(
        document,
        json!({"object": "response", "output": [message("msg_1", "pong")]})
    );
}

/// Copilot states its `copilot_usage` beside the terminal event's
/// `response`; the unary body states it at the top level.
#[test]
fn fields_beside_a_lifecycle_response_are_fields_of_the_document() {
    let usage = json!({"total_nano_aiu": 614_775_000});
    let mut terminal = lifecycle(
        "response.completed",
        envelope("completed", json!([message("msg_1", "pong")])),
    );
    terminal["copilot_usage"] = usage.clone();
    let document = rebuilt(&[
        lifecycle("response.created", envelope("in_progress", json!([]))),
        terminal,
    ]);
    assert_eq!(document["copilot_usage"], usage);
    assert_eq!(document.get("sequence_number"), None);
    assert_eq!(document.get("type"), None);
}

/// ChatGPT answers even a unary request with an event stream whose terminal
/// states no output, so both paths rebuild the output from the items the
/// stream finished.
#[tokio::test]
async fn chatgpt_unary_and_streamed_replies_rebuild_one_output() {
    let stream = sse(&[
        lifecycle("response.created", envelope("in_progress", json!([]))),
        item(
            "response.output_item.added",
            0,
            json!({"type": "function_call", "id": "fc_1", "arguments": ""}),
        ),
        json!({"type": "response.function_call_arguments.delta", "output_index": 0, "item_id": "fc_1", "delta": "{\"q\":1}"}),
        item("response.output_item.done", 0, call("fc_1", "{\"q\":1}")),
        lifecycle("response.completed", envelope("completed", json!([]))),
    ]);
    let wire = OpenAIConfig::with_key(&CHATGPT, "test-token").responses("gpt-5.4");
    let (unary, streamed) = raw_pair(wire, stream.clone(), stream)
        .await
        .unwrap_or_else(|error| panic!("the ChatGPT stream decodes: {error}"));
    assert_eq!(unary, streamed);
    assert_eq!(
        streamed,
        envelope("completed", json!([call("fc_1", "{\"q\":1}")]))
    );
}
