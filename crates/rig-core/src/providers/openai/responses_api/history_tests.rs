//! Replay rules the Responses wire holds by construction: each test is an
//! open review item of round 4, in the input shape that reproduced it.

use serde_json::{Value, json};

use crate::completion::{CompletionRequest, CompletionResponse, ToolDefinition};
use crate::message::{
    AssistantContent, AssistantMessage, CallId, Message, Origin, StopReason, ToolCall,
    ToolFunction, ToolName, ToolResultContent,
};
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

/// NEW-B and responses_new NEW-1: a call the provider named no call id
/// keeps its item, and replay writes the id its result answers into the
/// item's `call_id` slot, so the item and its result agree.
#[test]
fn a_call_without_a_call_id_keeps_its_item_and_pairs_with_its_result() {
    for call_id in [None, Some("")] {
        let mut item = json!({"type": "function_call", "id": "fc_1", "name": "f", "arguments": "{}", "status": "completed"});
        if let Some(call_id) = call_id {
            item["call_id"] = json!(call_id);
        }
        let response = whole(json!([item]));
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
            "call_id {call_id:?}: the provider's item goes back: {call}"
        );
        assert!(
            call["call_id"].as_str().is_some_and(|id| !id.is_empty())
                && call["call_id"] == output["call_id"],
            "call_id {call_id:?}: {call} answered by {output}"
        );
    }
}

/// NEW-C: a done message whose item states no content keeps the text that
/// streamed, and its item states that text, so it decodes back to its block.
#[test]
fn a_done_message_with_no_content_keeps_the_streamed_text() {
    let events = [
        json!({"type": "response.output_item.added", "output_index": 0,
               "item": {"type": "message", "id": "msg_1", "role": "assistant", "content": []}}),
        json!({"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": "the answer"}),
        json!({"type": "response.output_item.done", "output_index": 0,
               "item": {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed", "content": []}}),
        json!({"type": "response.completed", "response": {"id": "r", "status": "completed"}}),
    ];
    let response = streamed(&events);
    assert_eq!(response.text(), "the answer");
    let input = replayed(&response, &[]);
    assert_eq!(input[1]["id"], "msg_1");
    assert_eq!(input[1]["content"][0]["text"], "the answer", "{}", input[1]);
    let native = response.choice[0]
        .native_item()
        .cloned()
        .expect("the block keeps its item");
    assert_eq!(
        whole(json!([native])).text(),
        "the answer",
        "the item decodes back to its block"
    );
}

/// NEW-D: a result answering a call the provider stores is a custom output
/// when the request declares that tool custom, as pi decides by name.
#[test]
fn a_result_to_a_declared_custom_tool_is_a_custom_output() {
    let mut custom = ResponsesToolDefinition::hosted("custom");
    custom.name = "apply_patch".to_owned();
    let wire = wire().with_tool(custom);
    let mut request = CompletionRequest::from(vec![Message::tool_result(
        CallId::from_wire("call_1"),
        name("apply_patch"),
        "done",
    )]);
    request.additional_params = Some(json!({"previous_response_id": "resp_prev"}));
    let input = input_of(&wire, request);
    assert_eq!(
        input[0],
        json!({"type": "custom_tool_call_output", "call_id": "call_1", "output": "done"})
    );
}

/// NEW-E: one call id two foreign turns reuse goes out once per call, each
/// answered by its own result.
#[test]
fn a_call_id_two_foreign_turns_share_goes_out_distinct() {
    let origin = Origin::new("openai.chat", "llamacpp", "qwen3");
    let turn = |n: u64| {
        Message::Assistant(AssistantMessage {
            content: vec![AssistantContent::ToolCall(ToolCall::from_wire(
                "call_0",
                ToolFunction::new(name("f"), json!({"n": n})),
            ))],
            origin: Some(origin.clone()),
            stop: Some(StopReason::ToolUse),
        })
    };
    let result = |text: &str| Message::tool_result(CallId::from_wire("call_0"), name("f"), text);
    let history = vec![
        Message::user("q"),
        turn(1),
        result("one"),
        turn(2),
        result("two"),
    ];
    let input = input_of(
        &wire(),
        CompletionRequest::from(history).tools(vec![tool("f")]),
    );
    let ids: Vec<(&str, &str)> = input
        .iter()
        .filter_map(|item| Some((item["type"].as_str()?, item["call_id"].as_str()?)))
        .collect();
    assert_eq!(ids.len(), 4, "{input:?}");
    assert_ne!(ids[0].1, ids[2].1, "{ids:?}");
    assert_eq!(ids[0].1, ids[1].1, "{ids:?}");
    assert_eq!(ids[2].1, ids[3].1, "{ids:?}");
}

/// NEW-F: an item only the terminal states, at an output index the stream
/// gave another item, is kept: terminal items match by id before index.
#[test]
fn a_terminal_only_item_at_an_index_the_stream_used_is_kept() {
    let reasoning =
        json!({"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "e"});
    let message = json!({"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                         "content": [{"type": "output_text", "text": "hi", "annotations": []}]});
    let search = json!({"type": "web_search_call", "id": "ws_1", "status": "completed",
                        "action": {"type": "search", "query": "q"}});
    let events = [
        json!({"type": "response.output_item.done", "output_index": 0, "item": reasoning}),
        json!({"type": "response.output_item.done", "output_index": 1, "item": message}),
        json!({"type": "response.completed", "response": {"id": "r", "status": "completed", "output": [message, search]}}),
    ];
    let response = streamed(&events);
    assert_eq!(response.choice.len(), 3, "{:?}", response.choice);
    assert!(
        matches!(response.choice.last(), Some(AssistantContent::Opaque(opaque)) if opaque.item == search),
        "{:?}",
        response.choice
    );
}

/// NEW-G: reasoning never done before a call that is: the reasoning was
/// never complete, so the call after it goes back without its item id,
/// never as the `fc_` item OpenAI pairs with a reasoning item.
#[test]
fn a_call_after_reasoning_never_done_goes_back_without_its_item() {
    let mut events = streamed_call();
    events.remove(2);
    let response = streamed(&events);
    let input = replayed(&response, &["f"]);
    let call = input
        .iter()
        .find(|item| item["type"] == "function_call")
        .expect("the call goes back");
    assert!(
        input.iter().any(|item| item["type"] == "reasoning") || call.get("id").is_none(),
        "{input:?}"
    );
}

/// responses_new NEW-2 and NEW-3: a reasoning item goes back only with the
/// item after it, so a client-executed item that stays home, or text edited
/// to nothing, takes the reasoning before it along.
#[test]
fn reasoning_never_goes_back_without_the_item_after_it() {
    let reasoning = json!({"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "t"}],
                           "encrypted_content": "ENC"});
    let computer = json!({"type": "computer_call", "id": "cu_1", "call_id": "c1", "status": "completed",
                          "action": {"type": "screenshot"}, "pending_safety_checks": []});
    let message = json!({"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                         "content": [{"type": "output_text", "text": "hi", "annotations": []}]});
    let edited = |response: CompletionResponse| {
        let Some(Message::Assistant(mut turn)) = response.message() else {
            panic!("a turn");
        };
        for block in &mut turn.content {
            if let AssistantContent::Text(text) = block {
                text.text.clear();
            }
        }
        turn
    };
    for turn in [
        whole(json!([reasoning, computer])).message(),
        Some(Message::Assistant(edited(whole(json!([
            reasoning, message
        ]))))),
    ] {
        let turn = turn.expect("a turn");
        let input = input_of(
            &wire(),
            CompletionRequest::from(vec![Message::user("q"), turn, Message::user("n")]),
        );
        for (at, item) in input.iter().enumerate() {
            if item["type"] == "reasoning" {
                let next = input.get(at + 1);
                assert!(
                    next.is_some_and(|next| next
                        .get("role")
                        .is_none_or(|role| role == "assistant")),
                    "reasoning followed by {next:?}: {input:?}"
                );
            }
        }
    }
}

/// #2647 by rollback: a reply cut while reasoning is done but the item
/// after it is not keeps neither item, and one cut after both keeps both;
/// the terminal then backfills the ciphertext Azure states only there.
#[test]
fn reasoning_completes_with_the_item_after_it_and_takes_the_terminal_ciphertext() {
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
    let waiting = partial(5);
    assert!(
        waiting
            .choice
            .iter()
            .all(|block| block.native_item().is_none()),
        "reasoning waits for its call: {:?}",
        waiting.choice
    );
    let both = partial(7);
    assert!(
        both.choice
            .iter()
            .all(|block| block.native_item().is_some()),
        "both complete together: {:?}",
        both.choice
    );
    let last = events.len() - 1;
    events[last]["response"]["output"] = json!([
        {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "plan"}], "encrypted_content": "enc"},
    ]);
    let response = streamed(&events);
    assert_eq!(
        response.choice[0]
            .native_item()
            .map(|item| item["encrypted_content"].clone()),
        Some(json!("enc"))
    );
}

/// NEW-H: a completed reply with no output is an empty successful turn, in
/// both modes, as pi keeps it.
#[test]
fn an_empty_completed_reply_is_an_empty_successful_turn() {
    let stream = streamed(&[
        json!({"type": "response.completed", "response": {"id": "r", "status": "completed", "output": []}}),
    ]);
    let unary = whole(json!([]));
    for response in [stream, unary] {
        assert!(response.choice.is_empty());
        assert_eq!(response.stop(), StopReason::Stop);
    }
}
