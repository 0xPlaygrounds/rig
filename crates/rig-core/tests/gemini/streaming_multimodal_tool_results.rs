//! Gemini streaming regression for multimodal tool results in chat history.

use futures::StreamExt;
use rig_core::agent::MultiTurnStreamItem;
use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::completion::ToolDefinition;
use rig_core::message::{
    AssistantContent, DocumentSourceKind, ImageMediaType, Message, ToolResultContent, UserContent,
};
use rig_core::providers::gemini;
use rig_core::providers::gemini::completion::gemini_api_types::{
    AdditionalParameters, GenerationConfig,
};
use rig_core::streaming::StreamingPrompt;
use rig_core::tool::Tool;
use serde_json::json;

use crate::support::assert_nonempty_response;

const RED_PIXEL_PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==";
const MULTIMODAL_FUNCTION_RESPONSE_MODEL: &str = gemini::completion::GEMINI_3_FLASH_PREVIEW;

fn streaming_tool_params() -> serde_json::Value {
    serde_json::to_value(AdditionalParameters::default().with_config(GenerationConfig::default()))
        .expect("Gemini additional params should serialize")
}

#[derive(Debug)]
struct HybridImageTool;

#[derive(Debug, thiserror::Error)]
#[error("hybrid image tool error")]
struct HybridImageToolError;

impl Tool for HybridImageTool {
    const NAME: &'static str = "render_reference_image";
    type Error = HybridImageToolError;
    type Args = serde_json::Value;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Return a reference image the assistant must inspect before answering."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {},
                "required": [],
            }),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(json!({
            "response": {
                "instruction": "Use the image part to answer the user's question."
            },
            "parts": [
                {
                    "type": "image",
                    "data": RED_PIXEL_PNG_BASE64,
                    "mimeType": "image/png"
                }
            ]
        })
        .to_string())
    }
}

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY"]
async fn streaming_history_preserves_hybrid_tool_result_image_parts() {
    let client = gemini::Client::from_env().expect("client should build");
    let agent = client
        .agent(MULTIMODAL_FUNCTION_RESPONSE_MODEL)
        .preamble(
            "You are a precise assistant. Call `render_reference_image` exactly once before \
             answering. After the tool result arrives, do not call any more tools. Answer in one \
             short sentence.",
        )
        .tool(HybridImageTool)
        .additional_params(streaming_tool_params())
        .build();

    let empty_history: &[Message] = &[];
    let mut stream = agent
        .stream_prompt(
            "Use the tool once, then answer with the dominant color in the returned image.",
        )
        .with_history(empty_history)
        .multi_turn(4)
        .await;

    let mut final_response = None;
    let mut final_history = None;

    while let Some(item) = stream.next().await {
        match item.expect("streaming prompt should succeed") {
            MultiTurnStreamItem::FinalResponse(response) => {
                final_response = Some(response.response().to_owned());
                final_history = response.history().map(|history| history.to_vec());
                break;
            }
            MultiTurnStreamItem::StreamAssistantItem(_)
            | MultiTurnStreamItem::StreamUserItem(_) => {}
            _ => {}
        }
    }

    let final_response = final_response.expect("stream should yield a final response");
    assert_nonempty_response(&final_response);

    let history = final_history.expect("final response should include updated history");
    assert!(
        history.iter().any(|message| matches!(
            message,
            Message::Assistant { content, .. }
                if content.iter().any(|item| matches!(
                    item,
                    AssistantContent::ToolCall(tool_call)
                        if tool_call.function.name == HybridImageTool::NAME
                ))
        )),
        "history should retain the assistant tool call: {history:#?}"
    );

    let tool_result = history
        .iter()
        .find_map(|message| match message {
            Message::User { content } => content.iter().find_map(|item| match item {
                UserContent::ToolResult(tool_result) => Some(tool_result),
                _ => None,
            }),
            _ => None,
        })
        .expect("history should retain the tool result user message");

    assert_eq!(
        tool_result.content.len(),
        2,
        "hybrid tool results should round-trip as [text, image], not a single text blob: {tool_result:#?}"
    );

    let mut saw_instruction_text = false;
    let mut saw_image = false;

    for content in tool_result.content.iter() {
        match content {
            ToolResultContent::Text(text) => {
                saw_instruction_text = true;
                assert!(
                    text.text.contains("Use the image part"),
                    "tool result text should contain only the structured response payload: {text:?}"
                );
                assert!(
                    !text.text.contains(RED_PIXEL_PNG_BASE64),
                    "tool result text should not inline the image base64 payload: {text:?}"
                );
            }
            ToolResultContent::Image(image) => {
                saw_image = true;
                assert_eq!(image.media_type, Some(ImageMediaType::PNG));
                assert!(
                    matches!(image.data, DocumentSourceKind::Base64(ref data) if data == RED_PIXEL_PNG_BASE64),
                    "tool result image should preserve the base64 image payload: {image:?}"
                );
            }
        }
    }

    assert!(
        saw_instruction_text,
        "hybrid tool results should preserve the response text payload"
    );
    assert!(
        saw_image,
        "hybrid tool results should preserve the image payload as ToolResultContent::Image"
    );
}
