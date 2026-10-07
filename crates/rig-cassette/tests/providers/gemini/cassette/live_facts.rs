//! Live facts behind two Gemini decisions Rig makes without code:
//!
//! | # | Cell | Fact |
//! |---|------|------|
//! | 1 | `media_resolution/per_part_low` | The Developer API honours a part's own `mediaResolution`: the same image costs fewer image tokens at `MEDIA_RESOLUTION_LOW` than at the default. Rig sends no per-part setting on any wire; the vendored gRPC `Part` has no field for one. |
//! | 2 | `stream_function_call_arguments/developer_api` | A request that sets `streamFunctionCallArguments` still gets each call's arguments whole from the Developer API, never as `partialArgs`. Vertex AI documents `partialArgs` for the flag; Rig's Vertex wire is unary, and the Vertex streaming form is not verified here. |
//!
//! Cell 1 sends raw requests, through the session's client: Rig's images
//! carry no per-part settings. Every check of cell 2 runs on the recorded
//! exchanges before a recording is written.

use futures::StreamExt;
use rig::completion::{CompletionRequest, Effort, ToolDefinition};
use rig::message::{AssistantContent, ToolName};
use rig::providers::gemini::completion::GEMINI_3_FLASH_PREVIEW;
use serde_json::{Value, json};

use super::super::support::{with_gemini_cassette, with_gemini_checked_cassette};

/// A one-pixel PNG.
const PNG: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";

/// The image tokens a `generateContent` reply reports.
fn image_tokens(reply: &Value) -> u64 {
    reply["usageMetadata"]["promptTokensDetails"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|detail| detail["modality"] == "IMAGE")
        .and_then(|detail| detail["tokenCount"].as_u64())
        .unwrap_or_else(|| panic!("an image token count: {reply}"))
}

#[tokio::test]
async fn per_part_media_resolution() {
    with_gemini_cassette("media_resolution/per_part_low", |models| async move {
        let client = rig_test_support::cassettes::local_reqwest();
        let http = client.inner().expect("the session's reqwest client");
        let url = format!(
            "{}/v1beta/models/{GEMINI_3_FLASH_PREVIEW}:generateContent?key={}",
            models.config.base_url.trim_end_matches('/'),
            models.config.api_key.expose()
        );
        let mut tokens = Vec::new();
        for resolution in [None, Some("MEDIA_RESOLUTION_LOW")] {
            let mut image = json!({"inlineData": {"mimeType": "image/png", "data": PNG}});
            if let Some(level) = resolution {
                image["mediaResolution"] = json!({"level": level});
            }
            let body = json!({
                "contents": [{"role": "user", "parts": [image, {"text": "Say ok."}]}],
                "generationConfig": {"maxOutputTokens": 200}
            });
            let response = http
                .post(&url)
                .header("content-type", "application/json")
                .body(body.to_string())
                .send()
                .await
                .expect("the raw request is sent");
            assert_eq!(response.status().as_u16(), 200, "Gemini accepts the part");
            let reply: Value = response.json().await.expect("a JSON reply");
            tokens.push(image_tokens(&reply));
        }
        assert!(
            tokens[1] < tokens[0],
            "MEDIA_RESOLUTION_LOW costs fewer image tokens than the default: {tokens:?}"
        );
    })
    .await;
}

#[tokio::test]
async fn stream_function_call_arguments() {
    with_gemini_checked_cassette(
        "stream_function_call_arguments/developer_api",
        |models| async move {
            let request = CompletionRequest::new(
                "Call lookup with q set to a 40 word sentence about harbors.",
            )
            .tool(ToolDefinition::new(
                ToolName::new("lookup").expect("a tool name"),
                "Looks a sentence up.",
                json!({"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}),
            ))
            .max_tokens(4000)
            .reasoning(Effort::Low)
            .additional_params(json!({
                "toolConfig": {"functionCallingConfig": {
                    "mode": "ANY",
                    "streamFunctionCallArguments": true
                }}
            }));
            let mut stream = models
                .completion(GEMINI_3_FLASH_PREVIEW)
                .stream(request)
                .expect("the stream opens");
            while let Some(item) = stream.next().await {
                item.expect("no stream item is an error");
            }
            let response = stream.finish().await.expect("the stream ends cleanly");
            let call = response
                .choice
                .iter()
                .find_map(|block| match block {
                    AssistantContent::ToolCall(call) => Some(call.clone()),
                    _ => None,
                })
                .expect("the reply calls lookup");
            let q = call.function.arguments_value()["q"].clone();
            assert!(
                q.as_str().is_some_and(|q| q.split_whitespace().count() > 10),
                "the call's arguments arrive whole: {q}"
            );
        },
        |root, scenario| {
            let request = rig_cassette::http::recorded_json_request(root, "gemini", scenario);
            assert_eq!(
                request["toolConfig"]["functionCallingConfig"]["streamFunctionCallArguments"],
                true,
                "the recorded request asks for streamed arguments: {request}"
            );
            let frames = rig_cassette::http::recorded_sse_json_frames(root, "gemini", scenario);
            let text = serde_json::to_string(&frames).expect("frames serialize");
            assert!(
                !text.contains("partialArgs") && !text.contains("willContinue"),
                "the Developer API sends no partial arguments"
            );
            assert!(
                text.contains("functionCall"),
                "the recorded stream carries the call"
            );
        },
    )
    .await;
}
