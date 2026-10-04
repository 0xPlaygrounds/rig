//! The replay findings of the Ollama audit on its Chat dialect: arguments
//! are always an object, and images Ollama does not read are downgraded
//! rather than refused.

use serde_json::{Value, json};

use super::OllamaConfig;
use crate::completion::{CompletionRequest, Message, ToolDefinition};
use crate::message::{
    AssistantContent, AssistantMessage, CallId, DocumentSourceKind, Image, ImageMediaType, Origin,
    StopReason, ToolCall, ToolFunction, ToolName, ToolResultContent, UserContent,
};
use crate::providers::openai::wire::Chat;
use crate::test_utils::history::decode;
use crate::test_utils::json_body;
use crate::wire::{Mode, Operation, Wire, WireFrame};

const MODEL: &str = "qwen3:4b";

fn wire() -> Chat {
    OllamaConfig::new().completion(MODEL)
}

fn sent(history: Vec<Message>) -> Value {
    let wire = wire();
    // The request declares the tools, so calls and results stay calls.
    let mut request = CompletionRequest::new("next").tools(
        ["add", "shot"]
            .into_iter()
            .map(|tool| ToolDefinition::new(name(tool), "a tool", json!({"type": "object"})))
            .collect(),
    );
    request.chat_history = history;
    request.chat_history.push(Message::user("next"));
    let request = crate::operation::Completion::prepare(request, &wire.describe())
        .expect("the request prepares");
    json_body(&wire.encode(request, Mode::Unary).expect("encodes").request)
}

fn name(name: &str) -> ToolName {
    ToolName::new(name).expect("a tool name")
}

/// One `/v1/chat/completions` reply whose only call states `arguments`, in
/// the shape `ollama/tools/optional_argument.yaml` records.
fn reply(arguments: Value) -> ToolCall {
    let frame = WireFrame::Text(
        json!({"id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": MODEL,
            "system_fingerprint": "fp_ollama",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": "",
                "tool_calls": [{"id": "call_1", "index": 0, "type": "function",
                    "function": {"name": "add", "arguments": arguments}}]}}]})
        .to_string(),
    );
    let response = decode(&wire(), Mode::Unary, vec![frame]).expect("a call never fails");
    let calls: Vec<ToolCall> = response.tool_calls().cloned().collect();
    let [call] = calls.as_slice() else {
        panic!("the call is kept: {:?}", response.choice);
    };
    call.clone()
}

/// #1085, #2447, #2554 on Ollama: double-stringified arguments are an
/// object, malformed ones never fail the reply, and `null` ones go back as
/// an object.
#[test]
fn arguments_are_always_an_object() {
    assert_eq!(
        reply(json!("\"{\\\"x\\\":1}\"")).function.arguments_value(),
        json!({"x": 1})
    );
    let malformed = reply(json!("{\"x\": 1"));
    assert_eq!(malformed.function.arguments_value(), json!({"x": 1}));
    assert!(malformed.function.invalid_arguments.is_some());
    assert_eq!(reply(json!("null")).function.arguments_value(), json!({}));
    let turn = AssistantMessage {
        content: vec![AssistantContent::ToolCall(ToolCall::new(
            CallId::from_wire("call_1"),
            ToolFunction::new(name("add"), Value::Null),
        ))],
        origin: Some(Origin::new("openai.chat", "ollama", MODEL)),
        stop: Some(StopReason::ToolUse),
    };
    let body = sent(vec![Message::user("q"), Message::Assistant(turn)]);
    assert_eq!(
        body["messages"][1]["tool_calls"][0]["function"]["arguments"],
        "{}"
    );
}

/// chatA NEW-1, NEW-2, #2380 on Ollama: another model's assistant image and
/// a tool result's image reach Ollama downgraded, never refused, and an
/// image Ollama would have to fetch becomes a placeholder.
#[test]
fn images_ollama_does_not_read_are_downgraded() {
    let image = Image {
        data: DocumentSourceKind::base64("iVBORw0KGgo="),
        media_type: Some(ImageMediaType::PNG),
        ..Image::default()
    };
    let linked = Image {
        data: DocumentSourceKind::Url("https://example.com/a.png".to_owned()),
        media_type: Some(ImageMediaType::PNG),
        ..Image::default()
    };
    let call = ToolCall::new(
        CallId::from_wire("call_1"),
        ToolFunction::new(name("shot"), json!({})),
    );
    let history = vec![
        Message::User {
            content: vec![UserContent::text("look"), UserContent::Image(linked)],
        },
        Message::Assistant(AssistantMessage {
            content: vec![
                AssistantContent::Image(image.clone()),
                AssistantContent::ToolCall(call.clone()),
            ],
            origin: Some(Origin::new("gemini.generate_content", "gemini", "gemini-3")),
            stop: Some(StopReason::ToolUse),
        }),
        Message::User {
            content: vec![UserContent::ToolResult(
                call.result(vec![ToolResultContent::Image(image)]),
            )],
        },
    ];
    let body = sent(history);
    assert!(
        body["messages"][0]
            .to_string()
            .contains(crate::completion::history::IMAGE_UNSENDABLE),
        "{body}"
    );
    assert_eq!(
        body["messages"][2]["content"],
        crate::completion::history::TOOL_IMAGE_ATTACHED,
        "{body}"
    );
    assert_eq!(
        body["messages"][3]["content"][1]["image_url"]["url"], "data:image/png;base64,iVBORw0KGgo=",
        "{body}"
    );
}
