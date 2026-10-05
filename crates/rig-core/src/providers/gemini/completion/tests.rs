use crate::message;

use super::*;
use serde_json::json;

/// `message` as the content the REST wire sends Gemini 2.5 for it.
fn to_content(message: impl Into<message::Message>) -> Result<Value, EncodeError> {
    to_content_for("gemini-2.5-flash", message)
}

/// `message` as the content the REST wire sends `model` for it.
fn to_content_for(model: &str, message: impl Into<message::Message>) -> Result<Value, EncodeError> {
    contents_for(vec![message.into()], model)?
        .into_iter()
        .next()
        .ok_or_else(|| EncodeError::request("no content"))
}

/// The contents the REST wire sends `model` for `history`.
fn contents_for(history: Vec<message::Message>, model: &str) -> Result<Vec<Value>, EncodeError> {
    contents(history, &wire(model), model)
}

/// The request body the REST wire builds for `request` to `model`.
fn body_for(request: CompletionRequest, model: &str) -> Map<String, Value> {
    request_body(request, &wire(model), model).expect("the request encodes")
}

/// The parts of `content`.
fn parts(content: &Value) -> &[Value] {
    content["parts"].as_array().map_or(&[], Vec::as_slice)
}

#[tokio::test]
async fn test_generate_content_response_deserializes_without_candidates_or_response_id() {
    // Blocked prompt responses can omit default-valued proto fields, including
    // empty repeated `candidates` and empty string `responseId`.
    let body = json!({
        "promptFeedback": {
            "blockReason": "SAFETY"
        }
    });

    // A set `blockReason` is the provider's verdict on the prompt: the
    // error names it instead of reporting a generic missing-candidate parse
    // failure.
    let error = fold_unary("gemini-2.5-flash", body.to_string())
        .await
        .expect_err("a blocked prompt is an error");
    assert!(
        matches!(&error, ProviderError::ProviderResponse(response) if response.body.contains("blocked the prompt") && response.body.contains("SAFETY") && response.refusal && response.code.as_deref() == Some("SAFETY")),
        "{error}"
    );
    let report = crate::error::ErrorReport::from(&error);
    assert!(
        report.refusal,
        "the verdict is a refusal on the report: {report:?}"
    );
    assert!(!report.retryable, "{report:?}");
    assert_eq!(report.code.as_deref(), Some("SAFETY"));
}

#[tokio::test]
async fn test_blocked_prompt_error_carries_the_safety_ratings() {
    let error = fold_unary(
        "gemini-2.5-flash",
        json!({
            "promptFeedback": {
                "blockReason": "PROHIBITED_CONTENT",
                "safetyRatings": [
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "probability": "HIGH"}
                ]
            },
            "usageMetadata": {"promptTokenCount": 12, "totalTokenCount": 12}
        })
        .to_string(),
    )
    .await
    .expect_err("blocked");
    let message = error.to_string();
    assert!(message.contains("PROHIBITED_CONTENT"), "{message}");
    assert!(
        message.contains("HARM_CATEGORY_DANGEROUS_CONTENT"),
        "{message}"
    );
    assert!(message.contains("HIGH"), "{message}");
}

#[tokio::test]
async fn test_unknown_block_reason_reaches_the_error_verbatim() {
    let error = fold_unary(
        "gemini-2.5-flash",
        json!({"promptFeedback": {"blockReason": "SOMETHING_NEW"}}).to_string(),
    )
    .await
    .expect_err("blocked");
    assert!(
        error.to_string().contains("block_reason=SOMETHING_NEW"),
        "{error}"
    );
    // Unknown means unknown: not a verdict on the content, so retryable.
    assert!(error.is_retryable(), "{error:?}");
}

// A whole reply that names no block and carries no candidate never names
// the provider's end: it is truncated, not a block.
#[tokio::test]
async fn test_unspecified_block_reason_is_not_a_block() {
    let error = fold_unary(
        "gemini-2.5-flash",
        json!({"promptFeedback": {"blockReason": "BLOCK_REASON_UNSPECIFIED"}}).to_string(),
    )
    .await
    .expect_err("no candidates");
    assert!(matches!(error, ProviderError::Truncated), "{error}");
}

#[test]
fn test_unknown_finish_reason_round_trips_verbatim() {
    // A wire value this crate does not know keeps the provider's spelling.
    assert_eq!(
        map_google_finish_reason("FINISH_REASON_FUTURE"),
        crate::completion::FinishReason::Other("FINISH_REASON_FUTURE".to_string())
    );
}

#[test]
fn test_unknown_block_reason_deserializes_verbatim() {
    // Same contract for prompt feedback: a new block reason must not fail
    // the payload, and the spelling is preserved.
    let error = blocked_prompt_error(&json!({ "blockReason": "BLOCK_REASON_FUTURE" }))
        .expect("an unknown block reason is a block");
    assert!(
        matches!(&error, ProviderError::ProviderResponse(response) if response.code.as_deref() == Some("BLOCK_REASON_FUTURE")),
        "{error:?}"
    );
}

#[tokio::test]
async fn test_streaming_candidate_with_unknown_finish_reason_stays_parseable() {
    // A streamed terminal chunk with an unknown reason still produces the
    // terminal record, with the reason verbatim.
    let converted = streamed(
        "gemini-2.5-flash",
        concat!(
            r#"data: {"candidates":[{"content":{"parts":[{"text":"done"}],"role":"model"},"finishReason":"FINISH_REASON_FUTURE"}]}"#,
            "\r\n\r\n",
        ),
    )
    .await;

    assert_eq!(converted.text(), "done");
    assert_eq!(
        converted.finish_reason(),
        Some(crate::completion::FinishReason::Other(
            "FINISH_REASON_FUTURE".to_string()
        ))
    );
}

#[tokio::test]
async fn test_completion_response_carries_normalized_metadata() {
    let converted = unary(
        "gemini-2.5-flash",
        r#"{"responseId":"resp-meta","modelVersion":"gemini-2.0-flash-001","candidates":[{"content":{"parts":[{"text":"hi"}],"role":"model"},"finishReason":"MAX_TOKENS"}]}"#,
    )
    .await;

    assert_eq!(converted.provider(), PROVIDER_NAME);
    assert_eq!(converted.model(), Some("gemini-2.0-flash-001"));
    assert_eq!(converted.response_id(), Some("resp-meta"));
    assert_eq!(
        converted.finish_reason(),
        Some(crate::completion::FinishReason::Length)
    );
}

#[test]
fn test_tool_result_with_image_content() {
    // Test that a ToolResult with image content converts correctly to Gemini's Part format
    use crate::message::{
        DocumentSourceKind, Image, ImageMediaType, ToolResult, ToolResultContent,
    };

    // Create a tool result with both text and image content
    let tool_result = ToolResult { is_error: false,
        call: crate::message::CallId::from_wire("call-123"),
        name: crate::message::ToolName::new("test_tool".to_string()).expect("tool name"),
        content: vec![ToolResultContent::Text(message::Text::new(r#"{"status": "success"}"#.to_string())),ToolResultContent::Image(Image {
                data: DocumentSourceKind::Base64("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==".to_string()),
                media_type: Some(ImageMediaType::PNG),
                detail: None,
                native: None,
            })],
    };

    let user_content = message::UserContent::ToolResult(tool_result);
    let msg = message::Message::User {
        content: vec![user_content],
    };

    // Convert to Gemini Content
    let content =
        to_content_for("gemini-3-flash-preview", msg).expect("Should convert to Gemini Content");
    assert_eq!(content["role"], "user");
    assert_eq!(parts(&content).len(), 1);

    // Verify the part is a FunctionResponse with both response and parts
    let function_response = parts(&content)[0]
        .get("functionResponse")
        .expect("Expected FunctionResponse part");
    assert_eq!(function_response["name"], "test_tool");
    assert_eq!(function_response["id"], "call-123");

    // Check that response JSON is present
    assert_eq!(
        function_response["response"],
        json!({
            "result": r#"{"status": "success"}"#
        })
    );

    // Check that parts with image data are present
    let parts = function_response["parts"]
        .as_array()
        .expect("image parts are present");
    assert_eq!(parts.len(), 1);

    let inline_data = parts[0].get("inlineData").expect("inline image data");
    assert_eq!(inline_data["mimeType"], "image/png");
    assert!(
        inline_data["data"]
            .as_str()
            .is_some_and(|data| !data.is_empty())
    );
    assert!(inline_data.get("displayName").is_none());
}

#[test]
fn mixed_inline_images_and_text_keep_text_response_and_ordered_parts() {
    use crate::message::{ImageMediaType, ToolResult, ToolResultContent};

    let message = message::Message::User {
        content: vec![message::UserContent::ToolResult(ToolResult {
            is_error: false,
            call: crate::message::CallId::from_wire(""),
            name: crate::message::ToolName::new("ordered_tool".to_string()).expect("tool name"),
            content: vec![
                ToolResultContent::image_base64("first-image", Some(ImageMediaType::PNG), None),
                ToolResultContent::text("between-images"),
                ToolResultContent::image_base64("second-image", Some(ImageMediaType::JPEG), None),
            ],
        })],
    };

    let content = to_content(message).expect("tool result should convert");
    let response = parts(&content)[0]
        .get("functionResponse")
        .expect("expected a function response");

    assert_eq!(response["response"], json!({ "result": "between-images" }));

    let parts = response["parts"]
        .as_array()
        .expect("images should be inline parts");
    assert_eq!(parts.len(), 2);
    let first = parts[0].get("inlineData").expect("first inline image");
    assert_eq!(first["mimeType"], "image/png");
    assert_eq!(first["data"], "first-image");
    assert!(first.get("displayName").is_none());
    let second = parts[1].get("inlineData").expect("second inline image");
    assert_eq!(second["mimeType"], "image/jpeg");
    assert_eq!(second["data"], "second-image");
    assert!(second.get("displayName").is_none());
}

#[test]
fn mixed_inline_image_and_json_keep_structured_value_and_media_part() {
    use crate::message::{ImageMediaType, ToolResult, ToolResultContent};

    let message = message::Message::User {
        content: vec![message::UserContent::ToolResult(ToolResult {
            is_error: false,
            call: crate::message::CallId::from_wire(""),
            name: crate::message::ToolName::new("ordered_tool".to_string()).expect("tool name"),
            content: vec![
                ToolResultContent::json(json!({ "status": "ok" })),
                ToolResultContent::image_base64("image-data", Some(ImageMediaType::PNG), None),
            ],
        })],
    };

    let content = to_content(message).expect("tool result should convert");
    let response = parts(&content)[0]
        .get("functionResponse")
        .expect("expected a function response");

    assert_eq!(
        response["response"],
        json!({ "result": { "status": "ok" } })
    );
    let parts = response["parts"]
        .as_array()
        .expect("image should be an inline part");
    assert_eq!(parts.len(), 1);
    let inline_data = parts[0].get("inlineData").expect("inline image data");
    assert_eq!(inline_data["data"], "image-data");
    assert!(inline_data.get("displayName").is_none());
}

/// A function response carries JPEG, PNG and WEBP images only, so the
/// adapter replaces every other type there.
#[test]
fn tool_results_carry_only_jpeg_png_and_webp() {
    use crate::completion::{Media, Place};
    use crate::message::{DocumentSourceKind, Image, ImageMediaType};

    let image = |media_type| Image {
        data: DocumentSourceKind::base64("aW1hZ2U="),
        media_type: Some(media_type),
        ..Image::default()
    };
    for media_type in [
        ImageMediaType::JPEG,
        ImageMediaType::PNG,
        ImageMediaType::WEBP,
    ] {
        assert!(super::encodes(
            Media::Image(&image(media_type), Place::ToolResult),
            false
        ));
    }
    for media_type in [
        ImageMediaType::GIF,
        ImageMediaType::HEIC,
        ImageMediaType::HEIF,
        ImageMediaType::SVG,
    ] {
        assert!(
            !super::encodes(
                Media::Image(&image(media_type.clone()), Place::ToolResult),
                false
            ),
            "{media_type:?}"
        );
    }
    assert!(super::encodes(
        Media::Image(&image(ImageMediaType::HEIC), Place::User),
        false
    ));
}

#[test]
fn structured_json_refs_remain_literal_with_unreferenced_image_parts() {
    use crate::message::{ImageMediaType, ToolResult, ToolResultContent};

    let message = message::Message::User {
        content: vec![message::UserContent::ToolResult(ToolResult {
            is_error: false,
            call: crate::message::CallId::from_wire(""),
            name: crate::message::ToolName::new("collision_tool".to_string()).expect("tool name"),
            content: vec![
                ToolResultContent::json(json!({
                    "literal": {
                        "$ref": "tool_result_image_0"
                    }
                })),
                ToolResultContent::image_base64("image-data", Some(ImageMediaType::PNG), None),
            ],
        })],
    };

    let content = to_content(message).expect("tool result should convert");
    let response = parts(&content)[0]
        .get("functionResponse")
        .expect("expected a function response");

    assert_eq!(
        response["response"],
        json!({
            "result": {
                "literal": {
                    "$ref": "tool_result_image_0"
                }
            }
        })
    );
    assert!(
        response
            .pointer("/parts/0/inlineData/displayName")
            .is_none(),
        "{response}"
    );
}

#[test]
fn tool_result_literal_text_and_structured_json_remain_distinct() {
    use crate::message::{ToolResult, ToolResultContent};

    let cases = [
        (
            ToolResultContent::text(r#"{"status":"ok"}"#),
            json!({ "result": "{\"status\":\"ok\"}" }),
        ),
        (
            ToolResultContent::json(json!({ "status": "ok" })),
            json!({ "result": { "status": "ok" } }),
        ),
    ];

    for (tool_content, expected) in cases {
        let message = message::Message::User {
            content: vec![message::UserContent::ToolResult(ToolResult {
                is_error: false,
                call: crate::message::CallId::from_wire(""),
                name: crate::message::ToolName::new("test_tool".to_string()).expect("tool name"),
                content: vec![tool_content],
            })],
        };
        let content = to_content(message).expect("tool result should convert");

        let response = parts(&content)[0]
            .get("functionResponse")
            .expect("expected a function response");
        assert_eq!(response["response"], expected);
    }
}

/// A consumer echoing a minted `ToolCall::id` through `tool_result()` does
/// not put that handle on the wire of a model that takes no ids: the paired
/// functionCall omitted its id, and an asymmetric functionCall and
/// functionResponse id pair is rejected.
#[test]
fn echoed_minted_handle_never_reaches_the_function_response_id() {
    use crate::message::{CallId, ToolCall, ToolFunction, ToolResultContent};

    // An id-less wire: rig issued the id (Gemini REST issued none).
    let call = ToolCall::new(
        CallId::from_wire(""),
        ToolFunction::new(
            crate::message::ToolName::new("lookup".to_string()).expect("tool name"),
            json!({}),
        ),
    );

    let message = message::Message::User {
        content: vec![message::UserContent::ToolResult(message::ToolResult {
            is_error: false,
            call: call.id.clone(),
            name: call.function.name.clone(),
            content: vec![ToolResultContent::text("out")],
        })],
    };
    let content = to_content(message).expect("tool result should convert");
    let response = parts(&content)[0]
        .get("functionResponse")
        .expect("expected a function response");
    assert!(response.get("id").is_none(), "{response}");
}

#[test]
fn a_url_tool_result_image_is_sent_as_file_data() {
    use crate::message::{
        DocumentSourceKind, Image, ImageMediaType, ToolResult, ToolResultContent,
    };

    let tool_result = ToolResult {
        is_error: false,
        call: crate::message::CallId::from_wire(""),
        name: crate::message::ToolName::new("screenshot_tool".to_string()).expect("tool name"),
        content: vec![
            ToolResultContent::Image(Image {
                data: DocumentSourceKind::Url("https://example.com/image.png".to_string()),
                media_type: Some(ImageMediaType::PNG),
                detail: None,
                native: None,
            }),
            ToolResultContent::text("after-image"),
        ],
    };

    let content = to_content_for(
        "gemini-3-flash-preview",
        message::Message::User {
            content: vec![message::UserContent::ToolResult(tool_result)],
        },
    )
    .expect("a URL image encodes");
    let response = parts(&content)[0]
        .get("functionResponse")
        .unwrap_or_else(|| panic!("a function response: {content}"));
    assert_eq!(response["response"], json!({"result": "after-image"}));
    let file = response
        .pointer("/parts/0/fileData")
        .expect("the image is file data");
    assert_eq!(file["fileUri"], "https://example.com/image.png");
    assert_eq!(file["mimeType"], "image/png");
}

#[tokio::test]
async fn block_reasons_split_into_final_refusals_and_transient_blocks() {
    // `SAFETY`, `BLOCKLIST` and `PROHIBITED_CONTENT` judge the content and
    // are final. `OTHER` is Google's "blocked due to unknown reasons"; the
    // same prompt is answered on the next call, so it is retryable and it
    // is not a refusal. The classification is typed: the report's
    // `retryable` and `kind` say it, no caller reads the message. The block
    // arrives under a 200, which the driver keeps on the error for the
    // caller to see; a success status is no retry verdict, so the
    // decoder's own stands.
    for (reason, retryable) in [
        ("SAFETY", false),
        ("BLOCKLIST", false),
        ("PROHIBITED_CONTENT", false),
        ("OTHER", true),
        ("SOMETHING_NEW", true),
    ] {
        let error = fold_unary(
            "gemini-2.5-flash",
            json!({"promptFeedback": {"blockReason": reason}}).to_string(),
        )
        .await
        .expect_err(reason);
        assert_eq!(error.is_retryable(), retryable, "{reason}: {error:?}");
        let report = crate::error::ErrorReport::from(&error);
        assert_eq!(report.retryable, retryable, "{reason}: {report:?}");
        assert!(
            report.message.contains(&format!("block_reason={reason}")),
            "{reason}: {}",
            report.message
        );
        assert!(
            matches!(&error, ProviderError::ProviderResponse(response) if response.status == Some(http::StatusCode::OK) && response.refusal != retryable && response.code.as_deref() == Some(reason)),
            "{reason}: {error:?}"
        );
        assert_eq!(report.kind, crate::error::ErrorKind::ProviderResponse);
        assert_eq!(report.refusal, !retryable, "{reason}: {report:?}");
    }
}

// ── the GenerateContent wire ────────────────────────────────────────────
//
// Bodies below are pasted verbatim from committed cassettes, named at each
// constant. The point of the pairs is the property the wire model exists
// for: the unary reply and the streamed reply of the SAME turn, decoded by
// the SAME decoder, fold to the same answer.

use crate::test_utils::{MockStreamingClient, RecordingHttpClient};
use crate::wire::{Mode, Wire};
use futures::StreamExt;

/// `crates/rig-cassette/fixtures/cassettes/gemini/thought_text_matrix/blocking_keeps_a_trailing_thought_signature.yaml`
const SIGNED_UNARY: &str = r#"{"candidates":[{"content":{"parts":[{"text":"289","thoughtSignature":"c2lnbmF0dXJl"}],"role":"model"},"finishReason":"STOP","index":0}],"modelVersion":"gemini-3-flash-preview","responseId":"id_REDACTED_1","usageMetadata":{"candidatesTokenCount":2,"promptTokenCount":14,"promptTokensDetails":[{"modality":"TEXT","tokenCount":14}],"serviceTier":"standard","thoughtsTokenCount":43,"totalTokenCount":59}}"#;

/// `crates/rig-cassette/fixtures/cassettes/gemini/thought_text_matrix/streaming_twin_agrees_on_a_trailing_thought_signature.yaml`
/// — the same turn, streamed across two events, the signature riding a
/// trailing part that carries no `thought` flag.
const SIGNED_STREAM: &str = concat!(
    r#"data: {"candidates":[{"content":{"parts":[{"text":"289"}],"role":"model"},"index":0}],"modelVersion":"gemini-3-flash-preview","responseId":"id_REDACTED_1","usageMetadata":{"candidatesTokenCount":3,"promptTokenCount":14,"promptTokensDetails":[{"modality":"TEXT","tokenCount":14}],"serviceTier":"standard","thoughtsTokenCount":43,"totalTokenCount":60}}"#,
    "\r\n\r\n",
    r#"data: {"candidates":[{"content":{"parts":[{"text":"","thoughtSignature":"c2lnbmF0dXJl"}],"role":"model"},"finishReason":"STOP","index":0}],"modelVersion":"gemini-3-flash-preview","responseId":"id_REDACTED_1","usageMetadata":{"candidatesTokenCount":3,"promptTokenCount":14,"promptTokensDetails":[{"modality":"TEXT","tokenCount":14}],"serviceTier":"standard","thoughtsTokenCount":43,"totalTokenCount":60}}"#,
    "\r\n\r\n",
);

fn wire_request(prompt: &str) -> CompletionRequest {
    CompletionRequest::new(prompt)
}

fn wire(model: &str) -> GenerateContent {
    crate::providers::gemini::GeminiConfig::new("test-key").completion(model)
}

/// Fold one `generateContent` reply body through the bound wire, the way a
/// caller's `completion()` does — errors included.
async fn fold_unary(
    model: &str,
    body: impl Into<bytes::Bytes>,
) -> Result<crate::completion::CompletionResponse, ProviderError> {
    crate::driver::Model::new(wire(model), RecordingHttpClient::new(body))
        .call(wire_request("probe"))
        .await
}

async fn unary(model: &str, body: &'static str) -> crate::completion::CompletionResponse {
    fold_unary(model, body)
        .await
        .expect("the recorded unary reply decodes")
}

async fn streamed(model: &str, body: &'static str) -> crate::completion::CompletionResponse {
    let mut stream = crate::driver::Model::new(
        wire(model),
        MockStreamingClient {
            sse_bytes: bytes::Bytes::from_static(body.as_bytes()),
        },
    )
    .stream(wire_request("probe"))
    .expect("the stream opens");
    while let Some(item) = stream.next().await {
        item.expect("the recorded stream carries no in-band error");
    }
    stream
        .finish()
        .await
        .expect("the stream produced a terminal record")
}

/// Gemini hangs `thoughtSignature` on an answer part. The unary reply signs
/// its one answer part; the streamed twin sends the text, then an empty
/// signed part, which continues the same text block. Both decode to one
/// text whose provider item is the signed part, and both replay it as is.
#[tokio::test]
async fn a_trailing_thought_signature_joins_the_text_it_follows() {
    let buffered = unary("gemini-3-flash-preview", SIGNED_UNARY).await;
    let streamed = streamed("gemini-3-flash-preview", SIGNED_STREAM).await;
    assert_eq!(buffered.message(), streamed.message());
    let signed = json!({ "text": "289", "thoughtSignature": "c2lnbmF0dXJl" });
    let [message::AssistantContent::Text(text)] = buffered.choice.as_slice() else {
        panic!("one answer text: {:?}", buffered.choice);
    };
    assert_eq!(text.text, "289");
    assert_eq!(
        text.native.as_ref().map(|native| &native.item),
        Some(&signed)
    );

    let replayed = to_content(buffered.choice.clone()).expect("the turn replays");
    assert_eq!(replayed["parts"], json!([signed]));
}

/// Streaming is how a `generateContent` turn is delivered, not a different
/// operation: the unary and streamed twins report the same
/// `gen_ai.operation.name` and differ only in `gen_ai.request.stream`, as the
/// OpenTelemetry GenAI semantic conventions define them.
#[tokio::test]
async fn unary_and_streamed_twins_report_one_genai_operation() {
    let _isolation = crate::test_utils::scoped_tracing_subscriber_guard().await;
    let capture = crate::test_utils::TraceCapture::default();
    let _default = tracing::subscriber::set_default(capture.subscriber());
    unary("gemini-3-flash-preview", SIGNED_UNARY).await;
    streamed("gemini-3-flash-preview", SIGNED_STREAM).await;
    tracing::callsite::rebuild_interest_cache();
    capture.clear();

    unary("gemini-3-flash-preview", SIGNED_UNARY).await;
    streamed("gemini-3-flash-preview", SIGNED_STREAM).await;

    let spans = capture
        .spans()
        .into_iter()
        .filter(|span| span.target == "rig::completions")
        .map(|span| {
            (
                span.name,
                span.value("gen_ai.operation.name").cloned(),
                span.value("gen_ai.request.stream").cloned(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        spans,
        [
            (
                "generate_content",
                Some(json!("generate_content")),
                Some(json!(false))
            ),
            (
                "generate_content",
                Some(json!("generate_content")),
                Some(json!(true))
            ),
        ]
    );
}

/// The one request an `Encoded` carries.
fn sole(encoded: &crate::wire::Encoded) -> &http::Request<crate::wire::Body> {
    &encoded.request
}

/// A Gemini answer-text signature reaches only the model that issued it:
/// adapted for another Gemini model, the text replays without it.
#[test]
fn a_text_signature_reaches_no_other_model() {
    let signed = message::AssistantMessage {
        origin: Some(message::Origin::new(
            "gemini.generate_content",
            PROVIDER_NAME,
            "gemini-3-flash-preview",
        )),
        ..message::AssistantMessage::new(vec![
            message::AssistantContent::text("the answer")
                .with_native(json!({ "text": "the answer", "thoughtSignature": "c2lnbmVk" })),
        ])
    };
    let history = vec![message::Message::user("q"), signed.into()];
    for (model, part) in [
        (
            "gemini-3-flash-preview",
            json!({ "text": "the answer", "thoughtSignature": "c2lnbmVk" }),
        ),
        ("gemini-2.5-flash", json!({ "text": "the answer" })),
    ] {
        let adapted = crate::completion::adapt(&history, &wire(model));
        let contents = contents_for(adapted, model).expect("the history encodes");
        assert_eq!(contents[1]["parts"], json!([part]), "{model}");
    }
}

/// A signed answer text survives serde, which is how a history persists, and
/// still replays its part as Gemini sent it.
#[test]
fn a_signed_answer_text_round_trips_through_serde() {
    let message = message::Message::from(vec![
        message::AssistantContent::text("the answer")
            .with_native(json!({ "text": "the answer", "thoughtSignature": "c2lnbmVk" })),
    ]);
    let json = serde_json::to_string(&message).expect("the message serializes");
    let loaded: message::Message = serde_json::from_str(&json).expect("the message loads");
    assert_eq!(loaded, message);
    let content = to_content(loaded).expect("the turn replays");
    let [part] = parts(&content) else {
        panic!("one answer part: {content}");
    };
    assert_eq!(part["thoughtSignature"], "c2lnbmVk");
    assert_ne!(part.get("thought"), Some(&json!(true)));
}

/// A part rebuilt from canonical fields follows pi's rebuild: text alone, a
/// thought flag on reasoning, a call id only for a model that takes ids,
/// and nothing for redacted reasoning. Gemini 3 also gets Google's
/// placeholder signature on the call, which it requires. Blank text is
/// the adapter's to drop, not the encoder's.
#[test]
fn canonical_blocks_rebuild_as_pi_rebuilds_them() {
    let call = message::AssistantContent::tool_call(
        "call-1",
        message::ToolName::new("lookup").expect("tool name"),
        json!({ "q": 1 }),
    );
    let message = message::Message::from(vec![
        message::AssistantContent::reasoning("why"),
        message::AssistantContent::Reasoning(message::Reasoning {
            redacted: true,
            ..message::Reasoning::default()
        }),
        message::AssistantContent::text("answer"),
        call,
    ]);
    for (model, call) in [
        (
            "gemini-2.5-flash",
            json!({ "functionCall": { "name": "lookup", "args": { "q": 1 } } }),
        ),
        (
            "gemini-3-flash-preview",
            json!({
                "functionCall": { "name": "lookup", "args": { "q": 1 }, "id": "call-1" },
                "thoughtSignature": "skip_thought_signature_validator",
            }),
        ),
    ] {
        let contents = contents_for(vec![message.clone()], model).expect("the turn encodes");
        assert_eq!(
            contents[0]["parts"],
            json!([
                { "thought": true, "text": "why" },
                { "text": "answer" },
                call,
            ]),
            "{model}"
        );
    }
}

/// A `thoughtSignature` that is not base64 is left out of the replayed part:
/// Gemini rejects the whole request over one ("Invalid value at
/// 'contents[1].parts[0].thought_signature' (TYPE_BYTES), Base64 decoding
/// failed"). Every other field is sent as received.
#[test]
fn a_signature_that_is_not_base64_is_left_out() {
    let message = message::Message::from(vec![
        message::AssistantContent::text("a")
            .with_native(json!({ "text": "a", "thoughtSignature": "not base64!", "extra": 1 })),
        message::AssistantContent::text("b")
            .with_native(json!({ "text": "b", "thoughtSignature": "c2ln" })),
    ]);
    let contents = contents_for(vec![message], "gemini-2.5-flash").expect("the turn encodes");
    assert_eq!(
        contents[0]["parts"],
        json!([
            { "text": "a", "extra": 1 },
            { "text": "b", "thoughtSignature": "c2ln" },
        ])
    );
}

/// Call ids travel only to models that take them (pi's
/// `requiresToolCallId`), and another model's id is normalized for them.
#[test]
fn call_ids_follow_the_models_that_take_them() {
    for (model, takes) in [
        ("gemini-2.5-flash", false),
        ("gemini-2.0-flash", false),
        ("gemini-3-flash-preview", true),
        ("gemini-3.8-flash", true),
        ("gemini-live-3-flash", true),
        ("claude-sonnet-4-5", true),
        ("gpt-oss-120b", true),
        ("gemma-3-27b-it", false),
    ] {
        assert_eq!(requires_tool_call_id(model), takes, "{model}");
    }
    let foreign = format!("call.{}|x", "y".repeat(80));
    let normalized = normalize_tool_call_id("gemini-3-flash-preview", &foreign);
    assert_eq!(normalized.len(), 64);
    assert!(normalized.starts_with("call_yyy"));
    assert_eq!(
        normalize_tool_call_id("gemini-2.5-flash", &foreign),
        foreign
    );
}

/// `ThoughtReplay::CurrentTurn` leaves the signatures out of the turns
/// before the newest user text, and only the signatures.
#[test]
fn current_turn_replay_drops_only_finished_signatures() {
    let signed = |text: &str| {
        message::AssistantContent::text(text)
            .with_native(json!({ "text": text, "thoughtSignature": "c2ln" }))
    };
    let mut body = body_for(
        CompletionRequest::from(vec![
            message::Message::user("one"),
            message::Message::from(vec![signed("first")]),
            message::Message::user("two"),
            message::Message::from(vec![signed("second")]),
        ]),
        "gemini-2.5-flash",
    );
    let Some(Value::Array(contents)) = body.get_mut("contents") else {
        panic!("contents");
    };
    drop_finished_signatures(contents);
    assert_eq!(contents[1]["parts"], json!([{ "text": "first" }]));
    assert_eq!(
        contents[3]["parts"],
        json!([{ "text": "second", "thoughtSignature": "c2ln" }])
    );
}

/// Every part kind of the wire, an invented kind, and invented fields on
/// known kinds survive decode and same-model replay, whole and streamed.
#[test]
fn every_part_kind_survives_decode_and_replay() {
    use crate::test_utils::history::{assert_every_variant, assert_restated_agrees, decode};
    use crate::wire::WireFrame;

    let parts = vec![
        json!({ "thought": true, "text": "why", "thoughtSignature": "c2ln" }),
        json!({ "text": "plain", "futureField": 1 }),
        json!({
            "functionCall": { "name": "lookup", "args": { "q": 1 }, "id": "c1", "futureField": true },
            "thoughtSignature": "c2ln",
        }),
        json!({ "inlineData": { "mimeType": "image/png", "data": "iVBORw0KGgo=" } }),
        json!({ "functionResponse": { "name": "lookup", "response": { "ok": true } } }),
        json!({ "fileData": { "mimeType": "application/pdf", "fileUri": "gs://bucket/a.pdf" } }),
        json!({ "executableCode": { "language": "PYTHON", "code": "print(1)" } }),
        json!({ "codeExecutionResult": { "outcome": "OUTCOME_OK", "output": "1" } }),
        json!({ "futureKind": { "x": 1 } }),
    ];
    // The data fields of a Gemini `Part`, as the API documents them.
    const KINDS: [&str; 7] = [
        "text",
        "inlineData",
        "functionCall",
        "functionResponse",
        "fileData",
        "executableCode",
        "codeExecutionResult",
    ];
    let known: Vec<usize> = parts
        .iter()
        .take(8)
        .map(|part| {
            KINDS
                .iter()
                .position(|kind| part.get(*kind).is_some())
                .expect("a known part")
        })
        .collect();
    assert_every_variant(&known, |kind| *kind, KINDS.len());

    let end = json!({ "finishReason": "STOP", "index": 0 });
    let document = |parts: &[serde_json::Value], end: Option<&serde_json::Value>| {
        let mut candidate = json!({ "content": { "parts": parts, "role": "model" } });
        if let (Some(candidate), Some(serde_json::Value::Object(end))) =
            (candidate.as_object_mut(), end)
        {
            candidate.extend(end.clone());
        }
        WireFrame::Text(json!({ "candidates": [candidate], "responseId": "r" }).to_string())
    };
    let whole = vec![document(&parts, Some(&end))];
    let mut streamed: Vec<WireFrame> = parts
        .iter()
        .map(|part| document(std::slice::from_ref(part), None))
        .collect();
    streamed.push(document(&[], Some(&end)));

    let wire = wire("gemini-3-flash-preview");
    assert_restated_agrees(&wire, whole.clone(), streamed.clone());
    for (mode, frames) in [(Mode::Unary, whole), (Mode::Streaming, streamed)] {
        let response = decode(&wire, mode, frames).expect("the reply decodes");
        assert_eq!(response.choice.len(), parts.len(), "{mode:?}");
        let history = crate::completion::adapt(&[response.message().expect("a turn")], &wire);
        let replayed = contents_for(history, "gemini-3-flash-preview").expect("the turn replays");
        assert_eq!(replayed[0]["parts"], json!(parts), "{mode:?}");
    }
}

/// Grounding, URL context, safety ratings and citations stay with the turn
/// as its message-level native: the candidate without its content.
#[test]
#[ignore = "family C: the candidate's metadata moves to raw now that turns hold no message item"]
fn candidate_metadata_reaches_raw() {
    use crate::wire::WireFrame;

    let metadata = json!({
        "finishReason": "STOP",
        "index": 0,
        "safetyRatings": [{ "category": "HARM_CATEGORY_HARASSMENT", "probability": "NEGLIGIBLE" }],
        "citationMetadata": { "citationSources": [{ "uri": "https://example.com", "startIndex": 0, "endIndex": 4 }] },
        "groundingMetadata": { "webSearchQueries": ["rig"], "groundingChunks": [{ "web": { "uri": "https://example.com" } }] },
        "urlContextMetadata": { "urlMetadata": [{ "retrievedUrl": "https://example.com" }] },
    });
    let mut candidate = metadata.clone();
    candidate["content"] = json!({ "parts": [{ "text": "rig" }], "role": "model" });
    let frame = WireFrame::Text(json!({ "candidates": [candidate] }).to_string());
    let response =
        crate::test_utils::history::decode(&wire("gemini-2.5-flash"), Mode::Unary, [frame])
            .expect("the reply decodes");
    let candidate = response
        .raw
        .pointer("/candidates/0")
        .expect("the candidate");
    for (key, value) in metadata.as_object().expect("metadata is an object") {
        assert_eq!(candidate.get(key), Some(value), "{key}");
    }
}

/// A call rig issued the id for is spelled `tool-<n>` for a model that
/// takes ids, on the call and its response alike, so a history always
/// encodes to the same bytes.
#[test]
fn a_rig_issued_call_id_is_spelled_as_a_request_local_alias() {
    let call = message::ToolCall::from_wire(
        "",
        message::ToolFunction::new(
            message::ToolName::new("lookup").expect("tool name"),
            json!({}),
        ),
    );
    let history = vec![
        message::Message::from(call.clone()),
        message::Message::tool_results(vec![
            call.result(vec![message::ToolResultContent::text("out")]),
        ]),
    ];
    let contents = contents_for(history, "gemini-3-flash-preview").expect("the history encodes");
    assert_eq!(
        contents[0]["parts"][0]["functionCall"]["id"],
        json!("tool-0")
    );
    assert_eq!(
        contents[1]["parts"][0]["functionResponse"]["id"],
        json!("tool-0")
    );
}

/// The model a request addresses decides whether a foreign call id is
/// normalized, not the wire's own model.
#[test]
fn a_request_model_override_decides_call_id_normalization() {
    use crate::completion::ReplayTarget;

    let wire = GenerateContent::new(
        crate::providers::gemini::GeminiConfig::new("k"),
        "gemini-2.5-flash",
    );
    assert_eq!(
        wire.normalize_tool_call_id("a.b", "gemini-2.5-flash", None),
        "a.b"
    );
    assert_eq!(
        wire.normalize_tool_call_id("a.b", "gemini-3-pro-preview", None),
        "a_b"
    );
}

/// The body `request` sends to `model` once prepared, as the driver sends it.
fn prepared_body(model: &str, request: CompletionRequest) -> Value {
    use crate::wire::Operation;
    let wire = wire(model);
    let request = crate::operation::Completion::prepare(request, &wire.describe())
        .expect("the request prepares");
    let encoded = wire
        .encode(request, Mode::Unary)
        .expect("the request encodes");
    let crate::wire::Body::Bytes(bytes) = sole(&encoded).body() else {
        panic!("a JSON body");
    };
    serde_json::from_slice(bytes).expect("JSON")
}

#[tokio::test]
async fn an_unknown_traffic_type_or_modality_never_fails_the_reply() {
    let response = fold_unary(
        "gemini-2.5-flash",
        r#"{"candidates":[{"content":{"parts":[{"text":"kept"}],"role":"model"},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":2,"totalTokenCount":5,"trafficType":"ON_DEMAND_PRIORITY","promptTokensDetails":[{"modality":"HOLOGRAM","tokenCount":3}],"serviceTier":"flex"}}"#,
    )
    .await
    .expect("a usage label never fails the reply");
    assert_eq!(response.text(), "kept");
    assert_eq!(response.usage.total_tokens, Some(5));
}

/// A tool-result image as Gemini 2 and Gemini 3 get it: Gemini 3 reads it
/// inside the function response, Gemini 2 in a user message after it.
#[test]
fn tool_result_images_reach_gemini_2_in_a_following_user_message() {
    let call = message::ToolCall::new(
        message::CallId::from_wire("call_shot"),
        message::ToolFunction::new(
            message::ToolName::new("shot").expect("a tool name"),
            json!({}),
        ),
    );
    let image = message::Image {
        data: message::DocumentSourceKind::base64("aW1hZ2U="),
        media_type: Some(message::ImageMediaType::PNG),
        ..message::Image::default()
    };
    let history = vec![
        message::Message::user("look"),
        message::Message::Assistant(message::AssistantMessage::new(vec![
            message::AssistantContent::ToolCall(call.clone()),
        ])),
        message::Message::User {
            content: vec![message::UserContent::ToolResult(call.result(vec![
                message::ToolResultContent::text("see"),
                message::ToolResultContent::Image(image),
            ]))],
        },
    ];
    // A request that declares no tool gets its history's calls and results
    // as text, so this one declares the tool it replays.
    let request = |history: Vec<message::Message>| {
        let mut request =
            CompletionRequest::new("next").tools(vec![crate::completion::ToolDefinition {
                name: message::ToolName::new("shot").expect("a tool name"),
                description: "Take a screenshot".to_owned(),
                parameters: json!({ "type": "object", "properties": {} }),
            }]);
        request.chat_history = history;
        request
    };

    let body = prepared_body("gemini-2.5-flash", request(history.clone()));
    let contents = body["contents"].as_array().expect("contents");
    let response = contents
        .iter()
        .flat_map(|content| content["parts"].as_array().into_iter().flatten())
        .find_map(|part| part.get("functionResponse"))
        .expect("the result is sent");
    assert!(response.get("parts").is_none(), "{response}");
    assert!(
        contents.iter().any(|content| content["role"] == "user"
            && content["parts"]
                .as_array()
                .is_some_and(|parts| parts.iter().any(|part| part.get("inlineData").is_some()))),
        "the image follows in a user message: {body}"
    );

    let body = prepared_body("gemini-3-flash-preview", request(history));
    let response = body["contents"]
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(|content| content["parts"].as_array().into_iter().flatten())
        .find_map(|part| part.get("functionResponse"))
        .expect("the result is sent");
    assert!(response.get("parts").is_some(), "{response}");
}

/// A turn left with nothing to send, such as reasoning edited to blank, is
/// no content: Gemini rejects one with no parts.
#[test]
fn a_turn_with_nothing_to_send_is_no_content() {
    let turn = message::AssistantMessage::new(vec![message::AssistantContent::reasoning("")]);
    let contents = contents_for(
        vec![
            message::Message::user("q"),
            message::Message::Assistant(turn),
            message::Message::user("next"),
        ],
        "gemini-3-flash-preview",
    )
    .unwrap();
    assert!(
        contents.iter().all(|content| !parts(content).is_empty()),
        "{contents:?}"
    );
}

/// Gemini takes system text only in `systemInstruction` (round-5 F4):
/// `adapt` folds a later system message into the leading one, as pi does,
/// so the user turns it separated become one content.
#[test]
fn a_later_system_message_folds_into_the_instruction() {
    let history = vec![
        Message::system("s0"),
        Message::user("a"),
        Message::Assistant(crate::message::AssistantMessage::new(vec![
            crate::message::AssistantContent::text("ok"),
        ])),
        Message::user("b"),
        Message::system("later"),
        Message::user("c"),
    ];
    let body = prepared_body("gemini-2.5-flash", CompletionRequest::from(history));
    let roles: Vec<&str> = body["contents"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|content| content["role"].as_str())
        .collect();
    assert_eq!(roles, ["user", "model", "user"], "{}", body["contents"]);
    let instruction = body["systemInstruction"].to_string();
    assert!(
        instruction.contains("s0") && instruction.contains("later"),
        "{instruction}"
    );
}
