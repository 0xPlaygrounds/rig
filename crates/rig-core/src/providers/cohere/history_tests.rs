//! The replay findings of the Cohere audit, each as the test that closes
//! it: nothing the adapter hands over is refused, a vision model reads user
//! images, and a malformed call is kept.

use serde_json::{Value, json};

use super::{Chat, CohereConfig};
use crate::completion::{CompletionRequest, Message};
use crate::message::{
    AssistantContent, AssistantMessage, CallId, DocumentSourceKind, Image, ImageMediaType, Origin,
    StopReason, ToolCall, ToolFunction, ToolName, ToolResultContent, UserContent,
};
use crate::test_utils::history::{assert_every_variant, decode};
use crate::test_utils::json_body;
use crate::wire::{Mode, Operation, Wire, WireFrame};

fn wire(model: &str) -> Chat {
    CohereConfig::new("key").completion(model)
}

fn sent(model: &str, history: Vec<Message>) -> Value {
    let wire = wire(model);
    let mut request = CompletionRequest::new("next");
    request.chat_history = history;
    request.chat_history.push(Message::user("next"));
    let request = crate::operation::Completion::prepare(request, &wire.describe())
        .expect("the request prepares");
    json_body(&wire.encode(request, Mode::Unary).expect("encodes").request)
}

fn image() -> Image {
    Image {
        data: DocumentSourceKind::base64("iVBORw0KGgo="),
        media_type: Some(ImageMediaType::PNG),
        ..Image::default()
    }
}

/// chatA NEW-1, NEW-2, #2380, chatB NEW (placeholders): a text model gets
/// every image as a placeholder, and a vision model reads a user image and
/// a tool result's image moved to a user message.
#[test]
fn images_are_downgraded_for_what_each_model_reads() {
    let call = ToolCall::new(
        CallId::from_wire("call_1"),
        ToolFunction::new(ToolName::new("shot").expect("a tool name"), json!({})),
    );
    let history = vec![
        Message::User {
            content: vec![UserContent::text("look"), UserContent::Image(image())],
        },
        Message::Assistant(AssistantMessage {
            content: vec![
                AssistantContent::Image(image()),
                AssistantContent::ToolCall(call.clone()),
            ],
            origin: Some(Origin::new("gemini.generate_content", "gemini", "gemini-3")),
            stop: Some(StopReason::ToolUse),
        }),
        Message::User {
            content: vec![UserContent::ToolResult(
                call.result(vec![ToolResultContent::Image(image())]),
            )],
        },
    ];
    let text = sent(super::COMMAND_A_03_2025, history.clone());
    assert!(!text.to_string().contains("image_url"), "{text}");
    let vision = sent(super::COMMAND_A_VISION_07_2025, history);
    assert_eq!(
        vision["messages"][0]["content"],
        json!([{"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}]),
        "{vision}"
    );
    assert_eq!(
        vision["messages"][2]["content"],
        json!([{"type": "text", "text": crate::completion::history::TOOL_IMAGE_ATTACHED}]),
        "{vision}"
    );
    assert_eq!(
        vision["messages"][3]["content"][1]["type"], "image_url",
        "{vision}"
    );
}

/// #2447 on Cohere: a call whose arguments never parse is kept with what
/// they state, rather than dropped.
#[test]
fn a_malformed_call_is_kept() {
    let frame = WireFrame::Text(
        json!({"id": "m", "finish_reason": "TOOL_CALL", "message": {"role": "assistant",
            "tool_calls": [{"id": "call_1", "type": "function",
                "function": {"name": "add", "arguments": "{\"x\": 1"}}]}})
        .to_string(),
    );
    let response = decode(&wire(super::COMMAND_A_03_2025), Mode::Unary, vec![frame])
        .expect("a malformed call never fails the reply");
    let calls: Vec<&ToolCall> = response.tool_calls().collect();
    let [call] = calls.as_slice() else {
        panic!("the call is kept: {:?}", response.choice);
    };
    assert_eq!(call.function.arguments_value(), json!({"x": 1}));
    assert!(call.function.invalid_arguments.is_some());
}

/// Every content item kind has a sample, an invented one among them.
#[test]
fn every_content_item_has_a_sample() {
    let index = |block: &AssistantContent| match block {
        AssistantContent::Text(_) => 0,
        AssistantContent::Reasoning(_) => 1,
        AssistantContent::Opaque(_) => 2,
        AssistantContent::ToolCall(_) | AssistantContent::Image(_) => 3,
    };
    let content = json!([
        {"type": "text", "text": "a"},
        {"type": "thinking", "thinking": "b"},
        {"type": "x_rig_invented", "id": "x"},
    ]);
    let frame = WireFrame::Text(
        json!({"id": "m", "finish_reason": "COMPLETE",
            "message": {"role": "assistant", "content": content}})
        .to_string(),
    );
    let response = decode(&wire(super::COMMAND_A_03_2025), Mode::Unary, vec![frame])
        .expect("every item decodes");
    assert_every_variant(&response.choice, index, 3);
}
