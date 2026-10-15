use super::*;
use crate::wire::WireFrame;

fn image_generation_request(prompt: &str) -> ImageGenerationRequest {
    ImageGenerationRequest {
        prompt: prompt.to_string(),
        width: 1024,
        height: 1024,
        images: Vec::new(),
        additional_params: None,
    }
}

#[test]
fn request_body_uses_gemini_image_generation_shape() {
    let body = create_request_body(image_generation_request("Generate an image of an axolotl"))
        .expect("the body builds");
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

    let body = create_request_body(request).expect("the body builds");

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

/// The options set the size, swap the ratio the width and height picked, and
/// ask for text beside the images.
#[test]
fn image_options_set_size_ratio_and_text() {
    let mut request = image_generation_request("a banana");
    request.additional_params = Some(
        ImageOptions::new()
            .image_size(ImageSize::FiveTwelve)
            .aspect_ratio("21:9")
            .with_text()
            .params(),
    );
    let body = create_request_body(request).expect("the body builds");
    assert_eq!(
        body["generationConfig"],
        json!({
            "responseModalities": ["TEXT", "IMAGE"],
            "imageConfig": {"aspectRatio": "21:9", "imageSize": "512"}
        })
    );
    assert_eq!(
        ImageOptions::new().image_size(ImageSize::FourK).params(),
        json!({"generationConfig": {"imageConfig": {"imageSize": "4K"}}}),
        "without text, the modalities stay the wire's own"
    );
}

/// Width and height pick a ratio the API takes once reduced, and leave the
/// choice to the model otherwise.
#[test]
fn width_and_height_pick_a_ratio_the_api_takes() {
    for (width, height, ratio) in [
        (1920, 1080, Some("16:9")),
        (512, 2048, Some("1:4")),
        (2100, 900, Some("21:9")),
        (256, 256, Some("1:1")),
        (700, 301, None),
        (0, 1024, None),
    ] {
        assert_eq!(
            aspect_ratio(width, height).as_deref(),
            ratio,
            "{width}x{height}"
        );
    }
}

/// Reference images follow the prompt: bytes and base64 inline with their
/// type, a URL by reference.
#[test]
fn reference_images_follow_the_prompt() {
    use crate::message::ImageMediaType;

    let image = |data, media_type| Image {
        data,
        media_type,
        ..Image::default()
    };
    let mut request = image_generation_request("make it blue");
    request.images = vec![
        image(
            DocumentSourceKind::Raw(b"png".to_vec()),
            Some(ImageMediaType::PNG),
        ),
        image(
            DocumentSourceKind::Url("https://example.com/cat.jpeg".to_owned()),
            Some(ImageMediaType::JPEG),
        ),
    ];
    let body = create_request_body(request).expect("the body builds");
    assert_eq!(
        body["contents"][0]["parts"],
        json!([
            {"text": "make it blue"},
            {"inlineData": {"mimeType": "image/png", "data": "cG5n"}},
            {"fileData": {"mimeType": "image/jpeg", "fileUri": "https://example.com/cat.jpeg"}}
        ])
    );

    let mut untyped = image_generation_request("make it blue");
    untyped.images = vec![image(DocumentSourceKind::Base64("cG5n".to_owned()), None)];
    assert!(
        create_request_body(untyped).is_err(),
        "inline data needs its media type"
    );
}

/// Every text and image part comes back in order; thoughts, interim images
/// included, do not.
#[test]
fn parts_of_reads_text_and_images_in_order() {
    let reply = json!({"candidates": [{"content": {"parts": [
        {"text": "planning", "thought": true},
        {"inlineData": {"mimeType": "image/png", "data": "AA=="}, "thought": true},
        {"text": "Step one."},
        {"inlineData": {"mimeType": "image/jpeg", "data": "AQ=="}},
        {"inlineData": {"mimeType": "audio/wav", "data": "Ag=="}},
        {"inlineData": {"mimeType": "image/png"}},
        {"text": "Step two."},
        {"inlineData": {"mimeType": "image/png", "data": "Aw=="}}
    ]}}]});
    assert_eq!(
        parts_of(&reply).expect("the parts decode"),
        [
            ImagePart::Text("Step one.".to_owned()),
            ImagePart::Image {
                mime_type: "image/jpeg".to_owned(),
                data: vec![1]
            },
            ImagePart::Text("Step two.".to_owned()),
            ImagePart::Image {
                mime_type: "image/png".to_owned(),
                data: vec![3]
            },
        ]
    );
    assert_eq!(
        image_of(&reply).expect("an image").image,
        [1],
        "the response image is the first one"
    );
    let corrupt = json!({"candidates": [{"content": {"parts": [
        {"inlineData": {"mimeType": "image/png", "data": "not base64!"}}
    ]}}]});
    assert!(parts_of(&corrupt).is_err());
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

/// Gemini names an aspect ratio width first: a landscape 1024x768 is 4:3.
/// Sizes in no ratio Gemini names, or with no area, name none.
#[test]
fn the_aspect_ratio_names_the_width_first() {
    for ((width, height), ratio) in [
        ((1024, 1024), Some("1:1")),
        ((1024, 768), Some("4:3")),
        ((768, 1024), Some("3:4")),
        ((1920, 1080), Some("16:9")),
        ((1080, 1920), Some("9:16")),
        ((1000, 10), None),
        ((0, 512), None),
    ] {
        assert_eq!(
            aspect_ratio(width, height).as_deref(),
            ratio,
            "{width}x{height}"
        );
    }
}
