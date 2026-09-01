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
    };

    let err = response
        .normalize(super::super::completion::PROVIDER_NAME)
        .expect_err("text-only responses should fail");

    assert!(err.to_string().contains("did not include image data"));
}

#[test]
fn api_response_parsing_keeps_blocked_prompt_as_success() {
    let response: ApiResponse<GenerateContentResponse> = serde_json::from_value(json!({
        "promptFeedback": {
            "blockReason": "SAFETY"
        }
    }))
    .expect("blocked prompt response should deserialize");

    match response {
        ApiResponse::Ok(response) => assert!(response.candidates.is_empty()),
        ApiResponse::Err(err) => panic!("expected success envelope, got error: {err:?}"),
    }
}

#[tokio::test]
async fn image_generation_non_success_preserves_status_and_body() {
    use crate::client::image_generation::ImageGenerationClient;
    use crate::image_generation::ImageGenerationModel as _;
    use crate::test_utils::RecordingHttpClient;

    let body = r#"{"error":{"code":503,"message":"boom","status":"UNAVAILABLE"}}"#;
    let http_client =
        RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
    let client = Client::builder()
        .api_key("test-key")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.image_generation_model(GEMINI_2_5_FLASH_IMAGE);

    let error = model
        .image_generation(image_generation_request("draw a cat"))
        .await
        .expect_err("should fail with non-success status");

    assert!(matches!(error, ImageGenerationError::HttpError(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::SERVICE_UNAVAILABLE)
    );
    assert_eq!(error.provider_response_body(), Some(body));
}

#[tokio::test]
async fn image_generation_2xx_error_envelope_preserves_status_and_body() {
    use crate::client::image_generation::ImageGenerationClient;
    use crate::image_generation::ImageGenerationModel as _;
    use crate::test_utils::RecordingHttpClient;

    // 200 OK carrying Gemini's standard nested error envelope. The error
    // variant must be tried first because all identifying fields in
    // `GenerateContentResponse` can be omitted.
    let body = r#"{"error":{"code":503,"message":"boom","status":"UNAVAILABLE"}}"#;
    let http_client = RecordingHttpClient::new(body); // 200 OK
    let client = Client::builder()
        .api_key("test-key")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.image_generation_model(GEMINI_2_5_FLASH_IMAGE);

    let error = model
        .image_generation(image_generation_request("draw a cat"))
        .await
        .expect_err("should fail with provider error envelope");

    match &error {
        ImageGenerationError::ProviderResponse(stored) => {
            assert_eq!(stored.body, body);
            assert_eq!(stored.status, Some(http::StatusCode::OK));
        }
        other => panic!("expected ProviderResponse, got {other:?}"),
    }
}
