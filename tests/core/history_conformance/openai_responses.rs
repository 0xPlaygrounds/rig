//! The OpenAI Responses wire's history suite, and the fixture its dialects
//! (ChatGPT, Copilot's Responses route, xAI) share: replies are the API's
//! own response objects and SSE events, and the body is the JSON the wire
//! sends.

use rig_core::completion::CompletionRequest;
use rig_core::error::EncodeError;
use rig_core::operation::Completion;
use rig_core::providers::openai::OpenAIConfig;
use rig_core::providers::openai::responses_api::wire::Responses;
use rig_core::test_utils::history_conformance::{
    Ablation, Ending, HistoryFixture, Shape, http_body,
};
use rig_core::wire::{Encoded, Mode, Wire, WireFrame};
use serde_json::{Value, json};

/// One Responses wire or dialect: how to build it for a model, and the
/// models its replies come from.
pub struct ResponsesHistory<W> {
    /// The wire addressing a model.
    pub wire: fn(&str) -> W,
    /// The model the replies come from.
    pub model: &'static str,
    /// Another model on the same wire.
    pub other_model: &'static str,
    /// A model on the wire that reads no images.
    pub text_only_model: Option<&'static str>,
}

fn message(id: &str, text: &str, extra: Value) -> Value {
    let mut item = json!({
        "type": "message",
        "id": id,
        "role": "assistant",
        "status": "completed",
        "content": [{
            "type": "output_text",
            "text": text,
            "annotations": [{
                "type": "url_citation",
                "url": "https://example.com",
                "start_index": 0,
                "end_index": 1,
                "title": "Example",
            }],
            "logprobs": [],
        }],
    });
    if let (Some(item), Value::Object(extra)) = (item.as_object_mut(), extra) {
        item.extend(extra);
    }
    item
}

fn reasoning(id: &str, summary: &str) -> Value {
    json!({
        "type": "reasoning",
        "id": id,
        "summary": [{"type": "summary_text", "text": summary}],
        "encrypted_content": format!("ciphertext-of-{id}"),
    })
}

fn call(id: &str, call_id: &str, arguments: &str) -> Value {
    json!({
        "type": "function_call",
        "id": id,
        "call_id": call_id,
        "name": "lookup",
        "arguments": arguments,
        "status": "completed",
    })
}

fn output(shape: Shape) -> Vec<Value> {
    match shape {
        Shape::Rich => vec![
            reasoning("rs_1", "Plan the lookup."),
            message("msg_1", "Looking it up.", json!({"phase": "commentary"})),
            call("fc_1", "call_1", r#"{"q":"rig"}"#),
            json!({
                "type": "web_search_call",
                "id": "ws_1",
                "status": "completed",
                "action": {"type": "search", "query": "rig"},
            }),
        ],
        Shape::Interleaved => vec![
            reasoning("rs_1", "First."),
            message("msg_1", "Between.", json!({})),
            reasoning("rs_2", "Second."),
            call("fc_1", "call_1", r#"{"q":"rig"}"#),
        ],
        Shape::Unknown => vec![
            json!({"type": "x_rig_invented", "id": "x_1", "payload": {"n": 1}}),
            message("msg_1", "Noted.", json!({"x_rig_field": true})),
        ],
    }
}

/// The response object a reply of `output` ends with.
fn response(output: &[Value], status: Value) -> Value {
    let mut response = json!({
        "id": "resp_1",
        "object": "response",
        "created_at": 1_700_000_000,
        "model": "the-model",
        "output": output,
        "usage": {
            "input_tokens": 10,
            "input_tokens_details": {"cached_tokens": 2},
            "output_tokens": 20,
            "output_tokens_details": {"reasoning_tokens": 5},
            "total_tokens": 30,
        },
    });
    if let (Some(fields), Value::Object(status)) = (response.as_object_mut(), status) {
        fields.extend(status);
    }
    response
}

fn completed() -> Value {
    json!({"status": "completed"})
}

fn whole(output: &[Value], status: Value) -> Vec<WireFrame> {
    vec![WireFrame::Text(response(output, status).to_string())]
}

/// `item` as `output_item.added` states it: its text not yet streamed.
fn added(item: &Value) -> Value {
    let mut item = item.clone();
    if let Some(fields) = item.as_object_mut() {
        match fields.get("type").and_then(Value::as_str) {
            Some("message") => {
                fields.insert("content".to_owned(), json!([]));
                fields.insert("status".to_owned(), json!("in_progress"));
            }
            Some("reasoning") => {
                fields.insert("summary".to_owned(), json!([]));
                fields.shift_remove("encrypted_content");
            }
            Some("function_call") => {
                fields.insert("arguments".to_owned(), json!(""));
                fields.insert("status".to_owned(), json!("in_progress"));
            }
            _ => {
                fields.insert("status".to_owned(), json!("in_progress"));
            }
        }
    }
    item
}

/// The stream that restates `output`: each item added, its text streamed
/// in two deltas, then done, and the terminal response.
fn streamed(output: &[Value], status: Value) -> Vec<WireFrame> {
    let mut events = vec![json!({
        "type": "response.created",
        "sequence_number": 0,
        "response": {"id": "resp_1", "object": "response", "status": "in_progress", "output": []},
    })];
    for (index, item) in output.iter().enumerate() {
        events.push(json!({
            "type": "response.output_item.added",
            "output_index": index,
            "item": added(item),
        }));
        let (kind, text) = match item.get("type").and_then(Value::as_str) {
            Some("message") => (
                "response.output_text.delta",
                item.pointer("/content/0/text").and_then(Value::as_str),
            ),
            Some("reasoning") => (
                "response.reasoning_summary_text.delta",
                item.pointer("/summary/0/text").and_then(Value::as_str),
            ),
            Some("function_call") => (
                "response.function_call_arguments.delta",
                item.get("arguments").and_then(Value::as_str),
            ),
            _ => ("", None),
        };
        if let Some(text) = text {
            let middle = text.char_indices().nth(text.chars().count() / 2);
            let (head, tail) = text.split_at(middle.map_or(text.len(), |(at, _)| at));
            for delta in [head, tail] {
                events.push(json!({
                    "type": kind,
                    "output_index": index,
                    "item_id": item.get("id"),
                    "content_index": 0,
                    "summary_index": 0,
                    "delta": delta,
                }));
            }
        }
        events.push(json!({
            "type": "response.output_item.done",
            "output_index": index,
            "item": item,
        }));
    }
    let terminal = match status.get("status").and_then(Value::as_str) {
        Some("incomplete") => "response.incomplete",
        _ => "response.completed",
    };
    events.push(json!({"type": terminal, "response": response(output, status)}));
    events
        .into_iter()
        .map(|event| WireFrame::Text(event.to_string()))
        .collect()
}

/// One frame per event.
fn events(events: &[Value]) -> Vec<WireFrame> {
    events
        .iter()
        .map(|event| WireFrame::Text(event.to_string()))
        .collect()
}

fn reply(output: &[Value], status: Value, mode: Mode) -> Vec<WireFrame> {
    match mode {
        Mode::Unary => whole(output, status),
        Mode::Streaming => streamed(output, status),
    }
}

/// A rich whole reply as a provider echoes it, request config included.
fn document() -> Value {
    let mut document = response(&output(Shape::Rich), completed());
    if let Some(fields) = document.as_object_mut() {
        fields.extend(
            json!({
                "error": null,
                "incomplete_details": null,
                "instructions": "Be brief.",
                "max_output_tokens": null,
                "parallel_tool_calls": true,
                "previous_response_id": null,
                "reasoning": {"effort": "medium", "summary": "auto"},
                "service_tier": "default",
                "store": false,
                "temperature": 1.0,
                "text": {"format": {"type": "text"}, "verbosity": "medium"},
                "tool_choice": "auto",
                "tools": [{
                    "type": "function",
                    "name": "lookup",
                    "description": null,
                    "parameters": {"type": "object", "properties": {}},
                    "strict": null,
                }],
                "top_p": 1.0,
                "truncation": "disabled",
                "user": null,
                "metadata": {},
            })
            .as_object()
            .cloned()
            .unwrap_or_default(),
        );
    }
    document
}

impl<W> HistoryFixture for ResponsesHistory<W>
where
    W: Wire<Op = Completion, Payload = Encoded, Frame = WireFrame>,
{
    type Wire = W;

    fn wire(&self, model: &str) -> W {
        (self.wire)(model)
    }

    fn model(&self) -> &'static str {
        self.model
    }

    fn other_model(&self) -> &'static str {
        self.other_model
    }

    fn text_only_model(&self) -> Option<&'static str> {
        self.text_only_model
    }

    fn body(&self, wire: &W, request: CompletionRequest, mode: Mode) -> Result<Value, EncodeError> {
        http_body(&wire.encode(request, mode)?)
    }

    fn reply(&self, shape: Shape, mode: Mode) -> Option<Vec<WireFrame>> {
        Some(reply(&output(shape), completed(), mode))
    }

    fn call_reply(&self, arguments: &str, mode: Mode) -> Option<Vec<WireFrame>> {
        Some(reply(
            &[call("fc_1", "call_1", arguments)],
            completed(),
            mode,
        ))
    }

    /// Every status the API documents, every incomplete reason, and a body
    /// that states no status, which ended as the provider sent it. A
    /// streamed `response.incomplete` is incomplete whatever its status
    /// says, and a call announced and never done fails the turn.
    fn finishes(&self) -> Vec<(&'static str, Vec<WireFrame>, Ending)> {
        let text = [message("msg_1", "Done.", json!({}))];
        let calling = [call("fc_1", "call_1", "{}")];
        let incomplete = |reason: Value| json!({"status": "incomplete", "incomplete_details": {"reason": reason}});
        let failed = |status: &str| json!({"status": status, "error": {"code": "server_error", "message": "boom"}});
        vec![
            ("completed", whole(&text, completed()), Ending::Success),
            (
                "completed with a call",
                whole(&calling, completed()),
                Ending::Success,
            ),
            (
                "incomplete: max_output_tokens",
                whole(&text, incomplete(json!("max_output_tokens"))),
                Ending::Success,
            ),
            (
                "incomplete: content_filter",
                whole(&text, incomplete(json!("content_filter"))),
                Ending::Failure,
            ),
            (
                "incomplete: an unknown reason",
                whole(&text, incomplete(json!("max_tool_calls"))),
                Ending::Failure,
            ),
            (
                "incomplete without a reason",
                whole(
                    &text,
                    json!({"status": "incomplete", "incomplete_details": null}),
                ),
                Ending::Failure,
            ),
            ("failed", whole(&text, failed("failed")), Ending::Failure),
            (
                "cancelled",
                whole(&text, failed("cancelled")),
                Ending::Failure,
            ),
            (
                "queued",
                whole(&[], json!({"status": "queued"})),
                Ending::Failure,
            ),
            (
                "in_progress",
                whole(&text, json!({"status": "in_progress"})),
                Ending::Failure,
            ),
            (
                "an unknown status",
                whole(&text, json!({"status": "paused"})),
                Ending::Failure,
            ),
            ("no status", whole(&text, json!({})), Ending::Success),
            (
                "streamed incomplete at the output cap, without a status",
                events(&[
                    json!({"type": "response.output_item.done", "output_index": 0, "item": text[0]}),
                    json!({"type": "response.incomplete", "response": response(&text, json!({"incomplete_details": {"reason": "max_output_tokens"}}))}),
                ]),
                Ending::Success,
            ),
            (
                "streamed incomplete without a status or a reason",
                events(&[
                    json!({"type": "response.output_item.done", "output_index": 0, "item": text[0]}),
                    json!({"type": "response.incomplete", "response": response(&text, json!({}))}),
                ]),
                Ending::Failure,
            ),
            (
                "a call announced and never done",
                events(&[
                    json!({"type": "response.output_item.added", "output_index": 0, "item": added(&calling[0])}),
                    json!({"type": "response.function_call_arguments.delta", "output_index": 0, "delta": "{}"}),
                    json!({"type": "response.completed", "response": response(&[], completed())}),
                ]),
                Ending::Failure,
            ),
        ]
    }

    /// No field fails a reply: an item without its `type` is opaque, and
    /// every other field is read only when present.
    fn ablation(&self) -> Option<Ablation<WireFrame>> {
        Some(Ablation {
            document: document(),
            required: &[],
            frames: |document| vec![WireFrame::Text(document.to_string())],
        })
    }
}

/// The official OpenAI dialect.
pub fn openai(model: &str) -> Responses {
    Responses::new(OpenAIConfig::new("test-key"), model)
}

rig_core::history_conformance_suite! {
    wire: "openai_responses",
    fixture: ResponsesHistory {
        wire: openai,
        model: "gpt-5.4",
        other_model: "gpt-5.2",
        text_only_model: Some("o3-mini"),
    },
}
