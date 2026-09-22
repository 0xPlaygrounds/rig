use super::*;
use crate::providers::gemini::completion::gemini_api_types::{
    Blob, ContentCandidate, FinishReason, UsageMetadata,
};
use serde_json::json;

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
    let body = create_request_body(image_generation_request("Generate an image of an axolotl"))
        .expect("request should serialize");

    assert_eq!(
        generate_content_path(GEMINI_2_5_FLASH_IMAGE),
        "/v1beta/models/gemini-2.5-flash-image:generateContent"
    );
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

    let body = create_request_body(request).expect("request should serialize");

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
fn response_parsing_returns_first_non_thought_inline_image() {
    let response = GenerateContentResponse {
        candidates: vec![ContentCandidate {
            content: Some(Content {
                role: Some(Role::Model),
                parts: vec![
                    Part {
                        thought: Some(false),
                        thought_signature: None,
                        part: PartKind::Text("Here you go".to_string()),
                        additional_params: None,
                    },
                    Part {
                        thought: Some(true),
                        thought_signature: None,
                        part: PartKind::InlineData(Blob {
                            mime_type: "image/png".to_string(),
                            data: BASE64_STANDARD.encode("thought image"),
                        }),
                        additional_params: None,
                    },
                    Part {
                        thought: Some(false),
                        thought_signature: None,
                        part: PartKind::InlineData(Blob {
                            mime_type: "image/png".to_string(),
                            data: BASE64_STANDARD.encode("final image"),
                        }),
                        additional_params: None,
                    },
                ],
            }),
            finish_reason: Some(FinishReason::Stop),
            safety_ratings: None,
            citation_metadata: None,
            token_count: None,
            avg_logprobs: None,
            logprobs_result: None,
            index: None,
            finish_message: None,
        }],
        prompt_feedback: None,
        usage_metadata: Some(UsageMetadata {
            prompt_token_count: 1,
            cached_content_token_count: None,
            candidates_token_count: Some(1),
            total_token_count: 2,
            thoughts_token_count: None,
            prompt_tokens_details: None,
            cache_tokens_details: None,
            candidates_tokens_details: None,
            tool_use_prompt_token_count: None,
            tool_use_prompt_tokens_details: None,
            traffic_type: None,
        }),
        model_version: Some(GEMINI_2_5_FLASH_IMAGE.to_string()),
        response_id: "response-id".to_string(),
        error: None,
    };

    let parsed = response
        .normalize(super::super::completion::PROVIDER_NAME)
        .expect("response should contain an image");

    assert_eq!(parsed.image, b"final image");
}

#[test]
fn response_parsing_rejects_text_only_response() {
    let response = GenerateContentResponse {
        candidates: vec![ContentCandidate {
            content: Some(Content {
                role: Some(Role::Model),
                parts: vec![Part {
                    thought: Some(false),
                    thought_signature: None,
                    part: PartKind::Text("No image".to_string()),
                    additional_params: None,
                }],
            }),
            finish_reason: Some(FinishReason::Stop),
            safety_ratings: None,
            citation_metadata: None,
            token_count: None,
            avg_logprobs: None,
            logprobs_result: None,
            index: None,
            finish_message: None,
        }],
        prompt_feedback: None,
        usage_metadata: None,
        model_version: Some(GEMINI_2_5_FLASH_IMAGE.to_string()),
        response_id: "response-id".to_string(),
        error: None,
    };

    let err = response
        .normalize(super::super::completion::PROVIDER_NAME)
        .expect_err("text-only responses should fail");

    assert!(err.to_string().contains("did not include image data"));
}

/// A blocked prompt is a *successful* `generateContent` document: no
/// candidates, only `promptFeedback`. Every field of the response type is
/// optional or defaulted so that a reply carrying nothing but the block
/// still parses and the block can be reported, instead of the parse failing
/// and hiding the reason.
#[test]
fn a_blocked_prompt_reply_parses_as_a_candidate_less_response() {
    let response: GenerateContentResponse = serde_json::from_value(json!({
        "promptFeedback": {
            "blockReason": "SAFETY"
        }
    }))
    .expect("blocked prompt response should deserialize");

    assert!(response.candidates.is_empty());
    assert!(response.prompt_feedback.is_some());
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
        crate::providers::gemini::Gemini::new("test-key"),
        GEMINI_2_5_FLASH_IMAGE,
    );

    let mut driver = crate::driver::WireDriver::<ImageGeneration, _>::new(
        wire.decoder(crate::wire::Mode::Unary),
    );
    driver.push(WireFrame::Text(RECORDED_IMAGE_REPLY.to_string()));
    let decoded: Vec<_> = driver.drain().collect();

    let [Ok(response)] = decoded.as_slice() else {
        panic!("one whole reply decodes to one response, got {decoded:?}")
    };
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
