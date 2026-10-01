//! Provider items on the Responses wire: an output item this crate does not
//! model, `compaction` among them, is kept in place, a modelled item keeps
//! what its canonical form loses, both survive unary and streamed replies
//! alike, replay to this dialect as stated, and follow the staleness and
//! cross-dialect rules.

use serde_json::{Value, json};

use super::wire::Responses;
use crate::completion::{CompletionRequest, CompletionResponse};
use crate::message::{AssistantContent, Issuer, Message, NativeItem, Sealed, ToolName};
use crate::providers::openai::OpenAIConfig;
use crate::test_utils::{decode_reply, json_body};
use crate::wire::{Mode, Wire, WireFrame};

fn wire() -> Responses {
    Responses::new(OpenAIConfig::new("test-key"), "gpt-5.4")
}

fn response(output: &[Value]) -> Value {
    json!({
        "id": "resp_1", "object": "response", "created_at": 0, "status": "completed",
        "error": null, "incomplete_details": null, "instructions": null,
        "max_output_tokens": null, "model": "gpt-5.4", "usage": null, "output": output,
    })
}

/// The same reply as a stream states it: each item added, its payload by
/// delta, and done, then the terminal restating them all.
fn streamed(output: &[Value]) -> Vec<Value> {
    let mut sequence = 0;
    let mut next = || {
        sequence += 1;
        sequence
    };
    let mut frames = vec![
        json!({"type": "response.created", "sequence_number": next(),
        "response": response(&[])}),
    ];
    for (index, item) in output.iter().enumerate() {
        let mut added = item.clone();
        if item["type"] == "message" {
            added["content"] = json!([]);
        }
        frames.push(
            json!({"type": "response.output_item.added", "output_index": index,
            "sequence_number": next(), "item": added}),
        );
        if item["type"] == "message" {
            for (part, content) in item["content"].as_array().into_iter().flatten().enumerate() {
                frames.push(
                    json!({"type": "response.output_text.delta", "item_id": item["id"],
                    "output_index": index, "content_index": part, "sequence_number": next(),
                    "delta": content["text"]}),
                );
            }
        }
        if item["type"] == "function_call" {
            frames.push(json!({"type": "response.function_call_arguments.delta",
                "item_id": item["id"], "output_index": index, "sequence_number": next(),
                "delta": item["arguments"]}));
        }
        frames.push(
            json!({"type": "response.output_item.done", "output_index": index,
            "sequence_number": next(), "item": item}),
        );
    }
    frames.push(
        json!({"type": "response.completed", "sequence_number": next(),
        "response": response(output)}),
    );
    frames
}

fn decode(mode: Mode, frames: Vec<Value>) -> CompletionResponse {
    let wire = wire();
    let frames = frames
        .into_iter()
        .map(|frame| WireFrame::Text(frame.to_string()));
    decode_reply(
        &wire,
        &CompletionRequest::new("hello"),
        mode,
        frames,
        Value::Null,
    )
    .expect("the reply decodes")
}

/// The input items after the first a follow-up request replays `content`
/// as, with the reply's message id.
fn replayed(content: Vec<AssistantContent>, id: Option<&str>) -> Vec<Value> {
    let request = CompletionRequest::new("Thanks.").messages([
        Message::user("hello"),
        Message::Assistant {
            id: id.map(str::to_owned),
            content,
        },
    ]);
    let encoded = wire()
        .encode(request, Mode::Unary)
        .expect("the follow-up encodes");
    let input = json_body(&encoded.request)["input"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    // The first is the user's turn, the last the new prompt.
    input
        .iter()
        .skip(1)
        .take(input.len().saturating_sub(2))
        .cloned()
        .collect()
}

fn compaction() -> Value {
    json!({"type": "compaction", "id": "cmp_1", "encrypted_content": "opaque", "status": "completed"})
}

fn novel_output() -> Vec<Value> {
    vec![
        compaction(),
        json!({"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
               "phase": "commentary",
               "content": [{"type": "output_text", "text": "Checking.", "annotations": [],
                            "logprobs": []}]}),
        json!({"type": "hologram_call", "id": "hg_1", "status": "completed",
               "projection": {"frames": [1, 2, 3]}}),
        json!({"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "lookup",
               "arguments": "{\"record\":\"alpha\"}", "status": "completed",
               "namespace": "records"}),
    ]
}

/// Loss 2, and bar 1 and 4: `compaction` and an item type no version of
/// this crate knows are kept in place, unary and streamed replies of the
/// same output end with the same history, and replaying it sends every
/// item back as stated.
#[test]
fn compaction_and_novel_items_survive_both_modes_and_replay_as_stated() {
    let output = novel_output();
    let unary = decode(Mode::Unary, vec![response(&output)]);
    let stream = decode(Mode::Streaming, streamed(&output));
    assert_eq!(
        unary.choice, stream.choice,
        "both modes keep the same history"
    );

    let kinds: Vec<&str> = unary
        .choice
        .iter()
        .map(|part| match part {
            AssistantContent::Native(native) => native.value().kind().unwrap_or("?"),
            AssistantContent::Text(_) => "text",
            AssistantContent::ToolCall(_) => "tool_call",
            AssistantContent::Reasoning(_) | AssistantContent::Image(_) => "other",
        })
        .collect();
    assert_eq!(kinds, ["compaction", "text", "hologram_call", "tool_call"]);

    let without_status = |mut item: Value| {
        if let Some(item) = item.as_object_mut() {
            item.remove("status");
        }
        item
    };
    let message = {
        let mut message = output[1].clone();
        message["content"] = json!([{"type": "output_text", "text": "Checking."}]);
        message
    };
    assert_eq!(
        replayed(unary.choice, unary.message_id.as_deref()),
        [
            without_status(output[0].clone()),
            message,
            without_status(output[2].clone()),
            output[3].clone(),
        ],
        "every item replays as stated, its empty and default members aside"
    );
}

/// The staleness rule under the agent's block edits: a repaired call is
/// renamed, so its item no longer projects to it and it replays canonically;
/// an ignored call takes its item with it.
#[test]
fn edited_calls_replay_canonically() {
    let output = novel_output();
    let reply = decode(Mode::Unary, vec![response(&output)]);
    let mut choice = reply.choice;
    let Some(AssistantContent::ToolCall(call)) = choice.last_mut() else {
        panic!("a call");
    };
    assert!(call.native.is_some(), "its `namespace` rides it");
    call.function.name = ToolName::new("lookup_v2").expect("a name");
    let replay = replayed(choice.clone(), reply.message_id.as_deref());
    assert_eq!(
        replay.last(),
        Some(
            &json!({"type": "function_call", "id": "fc_1", "call_id": "call_1",
            "name": "lookup_v2", "arguments": "{\"record\":\"alpha\"}", "status": "completed"})
        )
    );

    choice.retain(|part| !matches!(part, AssistantContent::ToolCall(_)));
    let replay = replayed(choice, reply.message_id.as_deref());
    assert!(
        replay.iter().all(|item| item["type"] != "function_call"),
        "{replay:?}"
    );
}

/// The cross-dialect rule: another dialect's items are not sent, even
/// under an issuer this request opens; blocks they rode replay canonically.
#[test]
fn another_dialects_items_never_reach_this_wire() {
    let messages_item = |item: Value| {
        Sealed::new(
            Issuer::from("openai"),
            NativeItem::new("anthropic.messages", item),
        )
    };
    let content = vec![
        AssistantContent::Native(messages_item(
            json!({"type": "server_tool_use", "id": "srv_1"}),
        )),
        AssistantContent::Text(crate::message::Text {
            native: Some(messages_item(json!({"type": "text", "text": "Hi.",
                "citations": [{"type": "char_location"}]}))),
            ..crate::message::Text::new("Hi.")
        }),
    ];
    assert_eq!(
        replayed(content, None),
        [json!({"type": "message", "role": "assistant", "content": "Hi."})]
    );
}

/// A Responses item rides a block only while it says more than the block:
/// an ordinary reply keeps no provider item at all.
#[test]
fn ordinary_items_keep_no_provider_item() {
    let output = [
        json!({"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "e",
               "status": "completed"}),
        json!({"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
               "content": [{"type": "output_text", "text": "Hi.", "annotations": [],
                            "logprobs": []}]}),
        json!({"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "f",
               "arguments": "{\"a\": 1}", "status": "completed"}),
    ];
    let reply = decode(Mode::Unary, vec![response(&output)]);
    for part in &reply.choice {
        let native = match part {
            AssistantContent::Text(text) => text.native.as_ref(),
            AssistantContent::ToolCall(call) => call.native.as_ref(),
            AssistantContent::Reasoning(reasoning) => reasoning
                .open(reasoning.issuer())
                .and_then(|r| r.native.as_ref()),
            other => panic!("unexpected {other:?}"),
        };
        assert!(native.is_none(), "{part:?}");
    }
}
