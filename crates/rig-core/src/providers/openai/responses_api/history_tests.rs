//! Replay rules the Responses wire holds by construction: each test is an
//! open review item of round 4, in the input shape that reproduced it.

use serde_json::{Value, json};

use crate::completion::{CompletionRequest, CompletionResponse, ToolDefinition};
use crate::message::{AssistantContent, CallId, Message, ToolCall, ToolName, ToolResultContent};
use crate::operation::Completion;
use crate::providers::openai::OpenAIConfig;
use crate::wire::{Mode, Operation, Wire, WireFrame};

use super::ResponsesToolDefinition;
use super::wire::Responses;

fn wire() -> Responses {
    OpenAIConfig::new("key").responses("gpt-5.4")
}

fn name(name: &str) -> ToolName {
    ToolName::new(name).expect("a tool name")
}

fn tool(tool: &str) -> ToolDefinition {
    ToolDefinition {
        name: name(tool),
        description: "A tool".to_owned(),
        parameters: json!({"type": "object"}),
    }
}

fn frames(events: &[Value]) -> Vec<WireFrame> {
    events
        .iter()
        .map(|event| WireFrame::Text(event.to_string()))
        .collect()
}

fn body(output: Value) -> Value {
    json!({"id": "resp_1", "object": "response", "status": "completed", "model": "gpt-5.4", "output": output})
}

fn whole(output: Value) -> CompletionResponse {
    crate::test_utils::history::decode(
        &wire(),
        Mode::Unary,
        [WireFrame::Text(body(output).to_string())],
    )
    .expect("the body decodes")
}

fn streamed(events: &[Value]) -> CompletionResponse {
    crate::test_utils::history::decode(&wire(), Mode::Streaming, frames(events))
        .expect("the stream decodes")
}

/// The input `wire` sends for `request`, prepared as the driver prepares it.
fn input_of(wire: &Responses, request: CompletionRequest) -> Vec<Value> {
    let request = Completion::prepare(request, &wire.describe()).expect("the request prepares");
    let encoded = wire
        .encode(request, Mode::Unary)
        .expect("the request encodes");
    crate::test_utils::json_body(&encoded.request)["input"]
        .as_array()
        .cloned()
        .unwrap_or_default()
}

/// The input the same model gets for `response`'s turn, its calls answered.
fn replayed(response: &CompletionResponse, tools: &[&str]) -> Vec<Value> {
    let Some(turn) = response.message() else {
        panic!("the reply is a turn: {:?}", response.choice);
    };
    let results: Vec<_> = response
        .tool_calls()
        .map(|call| {
            crate::message::UserContent::ToolResult(
                call.result(vec![ToolResultContent::text("ok")]),
            )
        })
        .collect();
    let next = if results.is_empty() {
        Message::user("next")
    } else {
        Message::User { content: results }
    };
    let history = vec![Message::user("q"), turn, next];
    let request =
        CompletionRequest::from(history).tools(tools.iter().map(|name| tool(name)).collect());
    input_of(&wire(), request)
}

/// A reasoning item, then a call streamed as added, deltas and done.
fn streamed_call() -> Vec<Value> {
    vec![
        json!({"type": "response.output_item.added", "output_index": 0,
               "item": {"type": "reasoning", "id": "rs_1", "summary": []}}),
        json!({"type": "response.reasoning_summary_text.delta", "output_index": 0, "summary_index": 0, "delta": "plan"}),
        json!({"type": "response.output_item.done", "output_index": 0,
               "item": {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "plan"}],
                        "encrypted_content": "enc"}}),
        json!({"type": "response.output_item.added", "output_index": 1,
               "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "f", "arguments": ""}}),
        json!({"type": "response.function_call_arguments.delta", "output_index": 1, "delta": "{\"a\":"}),
        json!({"type": "response.function_call_arguments.delta", "output_index": 1, "delta": "1}"}),
        json!({"type": "response.output_item.done", "output_index": 1,
               "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "f",
                        "arguments": "{\"a\":1}", "status": "completed"}}),
        json!({"type": "response.completed", "response": {"id": "resp_1", "status": "completed", "output": []}}),
    ]
}

/// NEW-A and responses_new NEW-1: a done call item that leaves out its
/// arguments, name or call id keeps what its announcement and deltas
/// stated, as pi keeps the added block's id and name and parses
/// `item.arguments || partialJson`.
#[test]
fn a_done_call_keeps_what_was_announced_and_streamed() {
    for missing in ["arguments", "name", "call_id"] {
        let mut events = streamed_call();
        if let Some(item) = events[6]["item"].as_object_mut() {
            item.shift_remove(missing);
        }
        let response = streamed(&events);
        let calls: Vec<&ToolCall> = response.tool_calls().collect();
        assert_eq!(calls.len(), 1, "without `{missing}`: {:?}", response.choice);
        assert_eq!(calls[0].id.wire(), "call_1", "without `{missing}`");
        assert_eq!(calls[0].function.name.as_str(), "f", "without `{missing}`");
        assert_eq!(
            calls[0].function.arguments_value(),
            json!({"a": 1}),
            "without `{missing}`"
        );
        let input = replayed(&response, &["f"]);
        let call = input
            .iter()
            .find(|item| item["type"] == "function_call")
            .expect("the call goes back");
        let output = input
            .iter()
            .find(|item| item["type"] == "function_call_output")
            .expect("its result");
        assert_eq!(
            call["id"], "fc_1",
            "without `{missing}`: the item goes back"
        );
        assert_eq!(call["call_id"], output["call_id"], "without `{missing}`");
    }
}

/// #2647 by rollback: a reply cut before its terminal keeps no provider
/// item, whether or not the call after the reasoning is done; the terminal
/// keeps both and backfills the ciphertext Azure states only there.
#[test]
fn a_reply_cut_before_its_terminal_keeps_no_item_and_the_terminal_takes_the_ciphertext() {
    let mut events = streamed_call();
    if let Some(item) = events[2]["item"].as_object_mut() {
        item.shift_remove("encrypted_content");
    }
    let partial = |cut: usize| {
        crate::test_utils::history_conformance::partial(
            &wire(),
            Mode::Streaming,
            frames(&events[..cut]),
        )
        .0
    };
    assert!(!partial(7).choice.is_empty(), "both items ended");
    for cut in [3, 5, 7] {
        let cut_off = partial(cut);
        assert!(
            cut_off
                .choice
                .iter()
                .all(|block| block.native_item().is_none()),
            "cut {cut}: {:?}",
            cut_off.choice
        );
    }
    let last = events.len() - 1;
    events[last]["response"]["output"] = json!([
        {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "plan"}], "encrypted_content": "enc"},
    ]);
    let response = streamed(&events);
    assert!(
        response
            .choice
            .iter()
            .all(|block| block.native_item().is_some()),
        "{:?}",
        response.choice
    );
    assert_eq!(
        response.choice[0]
            .native_item()
            .map(|item| item["encrypted_content"].clone()),
        Some(json!("enc"))
    );
}

/// Round 5 generated-history finding 3: a stream that never sends output 0
/// and is cut before its terminal keeps no item of the call it streamed at
/// output 1, so the turn a runtime rolls back sends the call without an
/// item that would need the reasoning that never arrived.
#[test]
fn a_call_cut_off_after_a_missing_output_index_goes_back_without_its_item() {
    let call = json!({"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "f",
        "arguments": "{}", "status": "completed"});
    let mut added = call.clone();
    added["arguments"] = json!("");
    added["status"] = json!("in_progress");
    let events = [
        json!({"type": "response.created", "response": {"id": "resp_1", "status": "in_progress", "output": []}}),
        json!({"type": "response.output_item.added", "output_index": 1, "item": added}),
        json!({"type": "response.function_call_arguments.done", "output_index": 1, "item_id": "fc_1", "arguments": "{}"}),
        json!({"type": "response.output_item.done", "output_index": 1, "item": call}),
    ];
    let (response, ended) =
        crate::test_utils::history_conformance::partial(&wire(), Mode::Streaming, frames(&events));
    assert!(!ended, "no terminal arrived");
    assert!(
        response
            .choice
            .iter()
            .all(|block| block.native_item().is_none()),
        "{:?}",
        response.choice
    );
    let turn = response.continued(response.choice.clone());
    let results = turn
        .tool_calls()
        .map(|call| {
            crate::message::UserContent::ToolResult(
                call.result(vec![ToolResultContent::text("ok")]),
            )
        })
        .collect();
    let history = vec![
        Message::user("q"),
        Message::Assistant(turn),
        Message::User { content: results },
    ];
    let input = input_of(
        &wire(),
        CompletionRequest::from(history).tools(vec![tool("f")]),
    );
    assert!(
        input
            .iter()
            .any(|item| item["type"] == "function_call" && item["call_id"] == "call_1"),
        "{input:?}"
    );
    assert!(
        !input
            .iter()
            .any(|item| item.get("id") == Some(&json!("fc_1"))),
        "the call's item replays without output 0: {input:?}"
    );
}

fn reasoning(id: &str, ciphertext: Option<&str>) -> Value {
    let mut item = json!({"type": "reasoning", "id": id, "summary": [{"type": "summary_text", "text": "plan"}]});
    if let Some(ciphertext) = ciphertext {
        item["encrypted_content"] = json!(ciphertext);
    }
    item
}

fn message(id: &str, text: &str) -> Value {
    json!({"type": "message", "id": id, "role": "assistant", "status": "completed",
           "content": [{"type": "output_text", "text": text, "annotations": []}]})
}

fn function_call(id: &str, call_id: &str) -> Value {
    json!({"type": "function_call", "id": id, "call_id": call_id, "name": "f",
           "arguments": "{}", "status": "completed"})
}

fn kinds(response: &CompletionResponse) -> Vec<&'static str> {
    response
        .choice
        .iter()
        .map(|block| match block {
            AssistantContent::Text(_) => "text",
            AssistantContent::Reasoning(_) => "reasoning",
            AssistantContent::ToolCall(_) => "call",
            AssistantContent::Image(_) => "image",
            AssistantContent::Opaque(_) => "opaque",
        })
        .collect()
}

/// The input `wire` sends to continue `response`'s turn with its calls
/// answered, under `params`.
fn continued_on(wire: &Responses, response: &CompletionResponse, params: Value) -> Vec<Value> {
    let Some(turn) = response.message() else {
        panic!("the reply is a turn: {:?}", response.choice);
    };
    let results = response
        .tool_calls()
        .map(|call| {
            crate::message::UserContent::ToolResult(
                call.result(vec![ToolResultContent::text("ok")]),
            )
        })
        .collect();
    let history = vec![Message::user("q"), turn, Message::User { content: results }];
    let mut request = CompletionRequest::from(history).tools(vec![tool("f")]);
    request.additional_params = Some(params);
    input_of(wire, request)
}

/// A `store: false` request resolves no stored item: reasoning without its
/// ciphertext stays out, and the call after it goes without its item id.
#[test]
fn a_stateless_request_sends_no_reasoning_it_cannot_resolve() {
    let response = whole(json!([
        reasoning("rs_1", None),
        function_call("fc_1", "call_1")
    ]));
    let input = continued_on(&wire(), &response, json!({"store": false}));
    assert!(
        !input.iter().any(|item| item["type"] == "reasoning"),
        "{input:?}"
    );
    let call = input
        .iter()
        .find(|item| item["type"] == "function_call")
        .expect("the call goes back");
    assert!(call.get("id").is_none(), "{call}");
    assert_eq!(call["call_id"], "call_1");

    let stored = continued_on(&wire(), &response, json!({}));
    assert_eq!(stored[1]["id"], "rs_1", "{stored:?}");
    assert_eq!(stored[2]["id"], "fc_1", "{stored:?}");
}

/// A request in a stored `conversation` continues the provider's state,
/// which holds the calls its first results answer.
#[test]
fn a_conversation_continuation_keeps_the_results_it_answers() {
    let mut request = CompletionRequest::from(vec![Message::tool_result(
        CallId::from_wire("call_1"),
        name("f"),
        "done",
    )])
    .tools(vec![tool("f")]);
    request.additional_params = Some(json!({"conversation": "conv_1"}));
    let input = input_of(&wire(), request);
    assert_eq!(input[0]["type"], "function_call_output", "{input:?}");
    assert_eq!(input[0]["call_id"], "call_1");
}

/// A result takes the kind of the call it answers, even when the request
/// now declares a custom tool of that call's name.
#[test]
fn a_result_takes_the_kind_of_the_call_it_answers() {
    let mut custom = ResponsesToolDefinition::hosted("custom");
    custom.name = "f".to_owned();
    let response = whole(json!([function_call("fc_1", "call_1")]));
    let input = continued_on(&wire().with_tool(custom), &response, json!({}));
    let kinds: Vec<&str> = input
        .iter()
        .filter_map(|item| item["type"].as_str())
        .collect();
    assert_eq!(kinds, ["message", "function_call", "function_call_output"]);
}

/// A terminal whose output indices are shifted against the stream's
/// restates the streamed message, so its text is said once.
#[test]
fn a_terminal_shifted_against_the_stream_restates_its_items() {
    let events = [
        json!({"type": "response.output_text.delta", "output_index": 1, "content_index": 0, "delta": "hi"}),
        json!({"type": "response.completed", "response": {"id": "r", "status": "completed",
               "model": "gpt-5.4", "output": [message("msg_1", "hi")]}}),
    ];
    let response = streamed(&events);
    assert_eq!(response.text(), "hi", "{:?}", response.choice);
    assert_eq!(
        response.choice[0].native_item(),
        Some(&message("msg_1", "hi"))
    );
}

/// A hosted item only the terminal states takes its place between the
/// streamed items, as in the whole reply.
#[test]
fn a_terminal_only_hosted_item_keeps_its_place() {
    let search = json!({"type": "web_search_call", "id": "ws_1", "status": "completed",
                        "action": {"type": "search", "query": "q"}});
    let output = json!([
        reasoning("rs_1", Some("enc")),
        search,
        message("msg_1", "hi")
    ]);
    let events = [
        json!({"type": "response.output_item.done", "output_index": 0, "item": reasoning("rs_1", Some("enc"))}),
        json!({"type": "response.output_item.done", "output_index": 2, "item": message("msg_1", "hi")}),
        json!({"type": "response.completed", "response": {"id": "r", "status": "completed",
               "model": "gpt-5.4", "output": output.clone()}}),
    ];
    let whole = whole(output);
    assert_eq!(kinds(&streamed(&events)), kinds(&whole));
    assert_eq!(kinds(&whole), ["reasoning", "opaque", "text"]);
}

/// Reasoning streamed as raw text whose done item also states a summary
/// folds as the whole reply does.
#[test]
fn raw_reasoning_deltas_under_a_summary_fold_as_the_whole_reply() {
    let rs = json!({"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "sum"}],
                    "content": [{"type": "reasoning_text", "text": "raw"}], "encrypted_content": "enc"});
    let output = json!([rs, message("msg_1", "hi")]);
    let events = [
        json!({"type": "response.output_item.added", "output_index": 0, "item": {"type": "reasoning", "id": "rs_1", "summary": []}}),
        json!({"type": "response.reasoning_text.delta", "output_index": 0, "item_id": "rs_1", "content_index": 0, "delta": "raw"}),
        json!({"type": "response.reasoning_summary_text.delta", "output_index": 0, "item_id": "rs_1", "summary_index": 0, "delta": "sum"}),
        json!({"type": "response.output_item.done", "output_index": 0, "item": rs}),
        json!({"type": "response.output_item.added", "output_index": 1, "item": {"type": "message", "id": "msg_1", "role": "assistant", "content": []}}),
        json!({"type": "response.output_text.delta", "output_index": 1, "item_id": "msg_1", "content_index": 0, "delta": "hi"}),
        json!({"type": "response.output_item.done", "output_index": 1, "item": message("msg_1", "hi")}),
        json!({"type": "response.completed", "response": body(output.clone())}),
    ];
    let streamed = streamed(&events);
    let whole = whole(output);
    assert_eq!(streamed.choice, whole.choice);
}

/// A call with no name is dropped at decode, so the reasoning before it,
/// which goes only with that call, never goes back with the next item.
#[test]
fn reasoning_before_a_dropped_call_does_not_go_back() {
    let nameless = json!({"type": "function_call", "id": "fc_1", "call_id": "call_1",
                          "arguments": "{}", "status": "completed"});
    let response = whole(json!([
        reasoning("rs_1", Some("enc")),
        nameless,
        message("msg_1", "hi")
    ]));
    let input = replayed(&response, &["f"]);
    assert!(!input.iter().any(|item| item["id"] == "rs_1"), "{input:?}");
    assert!(
        input
            .iter()
            .any(|item| item["type"] == "message" && item["role"] == "assistant"),
        "{input:?}"
    );
}
