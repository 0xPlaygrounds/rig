//! The replay findings of the Ollama audit, each as the test that closes
//! it: the message is rebuilt from its blocks, arguments are always an
//! object, and nothing the adapter hands over is refused.

use serde_json::{Value, json};

use super::{Chat, OllamaConfig};
use crate::completion::{CompletionRequest, Message};
use crate::message::{
    AssistantContent, AssistantMessage, CallId, DocumentSourceKind, Image, ImageMediaType, Origin,
    Reasoning, StopReason, Text, ToolCall, ToolFunction, ToolName, ToolResultContent, UserContent,
};
use crate::test_utils::history::decode;
use crate::test_utils::json_body;
use crate::wire::{Mode, Operation, Wire, WireFrame};

fn wire() -> Chat {
    OllamaConfig::new().completion("qwen3:4b")
}

fn sent(history: Vec<Message>) -> Value {
    let wire = wire();
    let mut request = CompletionRequest::new("next");
    request.chat_history = history;
    request.chat_history.push(Message::user("next"));
    let request = crate::operation::Completion::prepare(request, &wire.describe())
        .expect("the request prepares");
    json_body(&wire.encode(request, Mode::Unary).expect("encodes").request)
}

fn name(name: &str) -> ToolName {
    ToolName::new(name).expect("a tool name")
}

fn same_model(content: Vec<AssistantContent>) -> AssistantMessage {
    AssistantMessage {
        content,
        origin: Some(Origin::new("ollama.chat", "ollama", "qwen3:4b")),
        stop: Some(StopReason::Stop),
        native: None,
    }
}

/// chatA NEW-5: every reasoning block reaches `thinking`, joined with a
/// newline, and text blocks join with nothing between them, as pi joins
/// them.
#[test]
fn the_rebuild_keeps_every_reasoning_block_and_joins_text_as_pi_does() {
    let turn = same_model(vec![
        AssistantContent::Reasoning(Reasoning::new("first thought")),
        AssistantContent::Text(Text::new("a")),
        AssistantContent::Reasoning(Reasoning::new("second thought")),
        AssistantContent::Text(Text::new("b")),
    ]);
    let body = sent(vec![Message::user("q"), Message::Assistant(turn)]);
    assert_eq!(
        body["messages"][1],
        json!({"role": "assistant", "content": "ab", "thinking": "first thought\nsecond thought"})
    );
}

/// #1085, #2447, #2554 on Ollama: double-stringified arguments are an
/// object, malformed ones and a nameless call never fail the reply, and
/// `null` ones go back as an object.
#[test]
fn arguments_are_always_an_object() {
    let reply = |arguments: Value| {
        let frame = WireFrame::Text(
            json!({"model": "qwen3:4b", "done": true, "done_reason": "stop", "message": {
                "role": "assistant", "content": "",
                "tool_calls": [{"function": {"name": "add", "arguments": arguments}},
                    {"function": {"arguments": {}}}]}})
            .to_string(),
        );
        let response = decode(&wire(), Mode::Unary, vec![frame]).expect("a call never fails");
        let calls: Vec<ToolCall> = response.tool_calls().cloned().collect();
        let [call] = calls.as_slice() else {
            panic!(
                "the named call is kept, the nameless one dropped: {:?}",
                response.choice
            );
        };
        call.clone()
    };
    assert_eq!(
        reply(json!("{\"x\":1}")).function.arguments_value(),
        json!({"x": 1})
    );
    let malformed = reply(json!("{\"x\": 1"));
    assert_eq!(malformed.function.arguments_value(), json!({"x": 1}));
    assert!(malformed.function.invalid_arguments.is_some());
    assert_eq!(reply(Value::Null).function.arguments_value(), json!({}));
    let turn = same_model(vec![AssistantContent::ToolCall(ToolCall::new(
        CallId::from_wire("call_1"),
        ToolFunction::new(name("add"), Value::Null),
    ))]);
    let body = sent(vec![Message::user("q"), Message::Assistant(turn)]);
    assert_eq!(
        body["messages"][1]["tool_calls"][0]["function"]["arguments"],
        json!({})
    );
}

/// chatA NEW-1, NEW-2, #2380 on Ollama: another model's assistant image and
/// a tool result's image reach Ollama downgraded, never refused.
#[test]
fn images_ollama_does_not_read_are_downgraded() {
    let image = Image {
        data: DocumentSourceKind::base64("iVBORw0KGgo="),
        media_type: Some(ImageMediaType::PNG),
        ..Image::default()
    };
    let call = ToolCall::new(
        CallId::from_wire("call_1"),
        ToolFunction::new(name("shot"), json!({})),
    );
    let history = vec![
        Message::user("look"),
        Message::Assistant(AssistantMessage {
            content: vec![
                AssistantContent::Image(image.clone()),
                AssistantContent::ToolCall(call.clone()),
            ],
            origin: Some(Origin::new("gemini.generate_content", "gemini", "gemini-3")),
            stop: Some(StopReason::ToolUse),
            native: None,
        }),
        Message::User {
            content: vec![UserContent::ToolResult(
                call.result(vec![ToolResultContent::Image(image)]),
            )],
        },
    ];
    let body = sent(history);
    assert_eq!(
        body["messages"][2]["content"],
        crate::completion::history::TOOL_IMAGE_ATTACHED,
        "{body}"
    );
    assert_eq!(
        body["messages"][3]["images"],
        json!(["iVBORw0KGgo="]),
        "{body}"
    );
}
