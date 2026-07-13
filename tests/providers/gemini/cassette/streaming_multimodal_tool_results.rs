//! Gemini streaming regressions for multimodal tool results in chat history.

use futures::StreamExt;
use rig::agent::MultiTurnStreamItem;
use rig::client::CompletionClient;
use rig::message::{DocumentSourceKind, ImageMediaType, Message, ToolResultContent};
use rig::providers::gemini;
use rig::providers::gemini::completion::gemini_api_types::{
    AdditionalParameters, GenerationConfig,
};
use rig::streaming::{StreamedUserContent, StreamingPrompt};
use rig::tool::Tool;
use serde_json::json;

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
async fn streaming_history_rejects_lossy_hybrid_tool_result_conversion() {
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

    let mut streamed_tool_result = None;
    let mut conversion_error = None;

    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult {
                tool_result,
                ..
            })) => streamed_tool_result = Some(tool_result),
            Ok(_) => {}
            Err(error) => {
                conversion_error = Some(error);
                break;
            }
        }
    }

    let error = conversion_error.expect("mixed tool-result conversion should stop the stream");
    assert!(
        error.to_string().contains("cannot preserve the order"),
        "unexpected conversion error: {error:?}"
    );

    let tool_result = streamed_tool_result.expect("the executed tool result should be streamed");

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
            ToolResultContent::Json { value } => {
                panic!("hybrid image fixture unexpectedly produced JSON content: {value}")
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

    })
    .await;
}
