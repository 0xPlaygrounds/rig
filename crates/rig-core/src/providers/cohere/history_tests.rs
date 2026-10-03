//! The replay findings of the Cohere audit on its Chat dialect: a text model
//! gets every image as a placeholder, and a vision model reads user images.

use serde_json::{Value, json};

use super::CohereConfig;
use crate::completion::{CompletionRequest, Message, ToolDefinition};
use crate::message::{
    AssistantContent, AssistantMessage, CallId, DocumentSourceKind, Image, ImageMediaType, Origin,
    StopReason, ToolCall, ToolFunction, ToolName, ToolResultContent, UserContent,
};
use crate::test_utils::json_body;
use crate::wire::{Mode, Operation, Wire};

fn sent(model: &str, history: Vec<Message>) -> Value {
    let wire = CohereConfig::new("key").completion(model);
    // The request declares the tool, so its calls and results stay calls.
    let mut request = CompletionRequest::new("next").tool(ToolDefinition::new(
        ToolName::new("shot").expect("a tool name"),
        "takes a screenshot",
        json!({"type": "object"}),
    ));
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
            {"type": "image_url",
                "image_url": {"url": "data:image/png;base64,iVBORw0KGgo=", "detail": "auto"}}]),
        "{vision}"
    );
    assert_eq!(
        vision["messages"][2]["content"],
        crate::completion::history::TOOL_IMAGE_ATTACHED,
        "{vision}"
    );
    assert_eq!(
        vision["messages"][3]["content"][1]["type"], "image_url",
        "{vision}"
    );
}
