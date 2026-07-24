//! Gemini streaming regression for multimodal tool results in chat history.

use futures::StreamExt;
use rig::OneOrMany;
use rig::agent::MultiTurnStreamItem;
use rig::message::{
    AssistantContent, DocumentSourceKind, ImageMediaType, Message, ToolResultContent, UserContent,
};
use rig::prelude::*;
use rig::providers::gemini;
use rig::providers::gemini::completion::gemini_api_types::{
    AdditionalParameters, GenerationConfig,
};
use rig::streaming::StreamingPrompt;
use rig::tool::{Tool, ToolOutput};
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
    type Output = ToolOutput;

    fn description(&self) -> String {
        "Return a reference image the assistant must inspect before answering.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {},
            "required": [],
        })
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let mut content = OneOrMany::one(ToolResultContent::json(json!({
            "instruction": "Use the image part to answer the user's question."
        })));
        content.push(ToolResultContent::image_base64(
            RED_PIXEL_PNG_BASE64,
            Some(ImageMediaType::PNG),
            None,
        ));
        Ok(ToolOutput::content(content))
    }
}

#[tokio::test]
async fn streaming_history_preserves_hybrid_tool_result_image_parts() {
    super::super::support::with_gemini_cassette("streaming_multimodal_tool_results/streaming_history_preserves_hybrid_tool_result_image_parts", |client| async move {
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
        .history(empty_history)
        .max_turns(4)
        .await;

    let mut final_response = None;
    let mut final_history = None;

    while let Some(item) = stream.next().await {
        match item.expect("streaming prompt should succeed") {
            MultiTurnStreamItem::FinalResponse(response) => {
                final_response = Some(response.output().to_owned());
                final_history = response.messages().map(|history| history.to_vec());
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
        "hybrid tool results should round-trip as [json, image], not a single text blob: {tool_result:#?}"
    );

    let mut saw_instruction_json = false;
    let mut saw_image = false;

    for content in tool_result.content.iter() {
        match content {
            ToolResultContent::Text(text) => {
                panic!("expected structured JSON and image blocks, got text: {text:?}")
            }
            ToolResultContent::Image(image) => {
                saw_image = true;
                assert_eq!(image.media_type, Some(ImageMediaType::PNG));
                assert!(
                    matches!(image.data, DocumentSourceKind::Base64(ref data) if data == RED_PIXEL_PNG_BASE64),
                    "tool result image should preserve the base64 image payload: {image:?}"
                );
            }
            ToolResultContent::Json { value } => {
                saw_instruction_json = true;
                assert_eq!(
                    value,
                    &json!({
                        "instruction": "Use the image part to answer the user's question."
                    }),
                    "tool result JSON should preserve the structured response payload"
                );
            }
        }
    }

    assert!(
        saw_instruction_json,
        "hybrid tool results should preserve the structured response payload"
    );
    assert!(
        saw_image,
        "hybrid tool results should preserve the image payload as ToolResultContent::Image"
    );

    })
    .await;
}
