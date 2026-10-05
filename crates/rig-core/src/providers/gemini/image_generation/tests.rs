use super::*;

fn image_generation_request(prompt: &str) -> ImageGenerationRequest {
    ImageGenerationRequest {
        prompt: prompt.to_string(),
        width: 1024,
        height: 1024,
        additional_params: None,
    }
}

#[test]
fn request_body_uses_gemini_image_generation_shape() {
    let body = create_request_body(image_generation_request("Generate an image of an axolotl"));
    let wire = Images::new(
        crate::providers::gemini::GeminiConfig::new("test-key"),
        GEMINI_2_5_FLASH_IMAGE,
    );
    let encoded = wire
        .encode(
            image_generation_request("Generate an image of an axolotl"),
            Mode::Unary,
        )
        .expect("the request encodes");

    assert_eq!(
        encoded.request.uri().path(),
        "/v1beta/models/gemini-2.5-flash-image:generateContent"
    );
    assert_eq!(crate::test_utils::json_body(&encoded.request), body);
    assert_eq!(body["contents"][0]["role"], "user");
    assert_eq!(
        body["contents"][0]["parts"][0]["text"],
        "Generate an image of an axolotl"
    );
    assert_eq!(
        body["generationConfig"]["responseModalities"],
        json!(["IMAGE"])
    );
    assert_eq!(
        body["generationConfig"]["imageConfig"]["aspectRatio"],
        "1:1"
    );
}

#[test]
fn request_body_allows_additional_params_to_override_image_config() {
    let mut request = image_generation_request("Generate an image of an axolotl");
    request.additional_params = Some(json!({
        "generationConfig": {
            "imageConfig": {
                "aspectRatio": "16:9",
                "imageSize": "2K"
            }
        }
    }));

    let body = create_request_body(request);

    assert_eq!(
        body["generationConfig"]["imageConfig"]["aspectRatio"],
        "16:9"
    );
    assert_eq!(body["generationConfig"]["imageConfig"]["imageSize"], "2K");
    assert_eq!(
        body["generationConfig"]["responseModalities"],
        json!(["IMAGE"])
    );
}

#[test]
fn response_parsing_rejects_text_only_response() {
    let reply = json!({
        "candidates": [{
            "content": { "role": "model", "parts": [{ "thought": false, "text": "No image" }] },
            "finishReason": "STOP",
        }],
        "modelVersion": GEMINI_2_5_FLASH_IMAGE,
        "responseId": "response-id",
    });

    let err = image_of(&reply).expect_err("text-only responses should fail");

    assert!(err.to_string().contains("did not include image data"));
}

/// A blocked prompt is a *successful* `generateContent` document: no
/// candidates, only `promptFeedback`. The wire recognizes it as a reply and
/// reports a response error, instead of failing to read it.
#[test]
fn a_blocked_prompt_reply_decodes_to_a_response_error() {
    let wire = Images::new(
        crate::providers::gemini::GeminiConfig::new("test-key"),
        GEMINI_2_5_FLASH_IMAGE,
    );

    let err = crate::test_utils::decode_reply(
        &wire,
        &image_generation_request("a banana"),
        crate::wire::Mode::Unary,
        [WireFrame::Text(
            json!({ "promptFeedback": { "blockReason": "SAFETY" } }).to_string(),
        )],
        serde_json::Value::Null,
    )
    .expect_err("a blocked prompt carries no image");

    assert!(
        matches!(&err, ProviderError::Response(message) if message.contains("did not include image data")),
        "{err:?}"
    );
}

/// The 200 reply recorded in
/// `crates/rig-cassette/fixtures/cassettes/gemini/image_generation/nano_banana_image_generation_smoke.yaml`,
/// verbatim except for `inlineData.data`: the recorded PNG is 1 MB of base64,
/// so only its first four base64 groups — the PNG signature and the start of
/// the `IHDR` chunk — are kept here.
const RECORDED_IMAGE_REPLY: &str = r#"{"candidates":[{"content":{"parts":[{"inlineData":{"data":"iVBORw0KGgoAAAANSUhEUgAA","mimeType":"image/png"}}],"role":"model"},"finishReason":"STOP","index":0}],"modelVersion":"gemini-2.5-flash-image","responseId":"id_REDACTED_1","usageMetadata":{"candidatesTokenCount":1290,"candidatesTokensDetails":[{"modality":"IMAGE","tokenCount":1290}],"promptTokenCount":15,"promptTokensDetails":[{"modality":"TEXT","tokenCount":15}],"serviceTier":"standard","totalTokenCount":1305}}"#;

#[test]
fn the_wire_decodes_a_recorded_image_reply() {
    let wire = Images::new(
        crate::providers::gemini::GeminiConfig::new("test-key"),
        GEMINI_2_5_FLASH_IMAGE,
    );

    let response = crate::test_utils::decode_reply(
        &wire,
        &image_generation_request("a banana"),
        crate::wire::Mode::Unary,
        [WireFrame::Text(RECORDED_IMAGE_REPLY.to_string())],
        serde_json::Value::Null,
    )
    .expect("one whole reply decodes to one response");
    // The base64 the cassette carries, decoded: `\x89PNG\r\n\x1a\n` then the
    // 13-byte-length `IHDR` chunk header.
    assert_eq!(
        response.image,
        [
            0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, b'I', b'H',
            b'D', b'R', 0x00, 0x00
        ]
    );
    assert_eq!(response.provider, super::super::completion::PROVIDER_NAME);
    assert_eq!(response.model.as_deref(), Some("gemini-2.5-flash-image"));
    assert_eq!(response.response_id.as_deref(), Some("id_REDACTED_1"));
    assert_eq!(response.usage.input_tokens, Some(15));
    assert_eq!(response.usage.output_tokens, Some(1290));
    assert_eq!(response.usage.total_tokens, Some(1305));
}
