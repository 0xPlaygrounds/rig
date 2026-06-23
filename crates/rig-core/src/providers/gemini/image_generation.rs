//! Gemini image generation support.

use super::client::{ApiResponse, Client};
use super::completion::gemini_api_types::{
    Content, GenerateContentRequest, GenerateContentResponse, GenerationConfig, ImageConfig, Part,
    PartKind, ResponseModality, Role,
};
use crate::http_client::HttpClientExt;
use crate::image_generation::{ImageGenerationError, ImageGenerationRequest};
use crate::{http_client, image_generation};
use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use serde_json::Value;

/// `gemini-2.5-flash-image` image generation model, commonly referred to as Nano Banana.
pub const GEMINI_2_5_FLASH_IMAGE: &str = super::completion::GEMINI_2_5_FLASH_IMAGE;

/// Gemini image generation model.
#[derive(Clone)]
pub struct ImageGenerationModel<T = reqwest::Client> {
    client: Client<T>,
    /// Name of the model, for example [`GEMINI_2_5_FLASH_IMAGE`].
    pub model: String,
}

impl<T> ImageGenerationModel<T> {
    pub(crate) fn new(client: Client<T>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }
}

impl TryFrom<GenerateContentResponse>
    for image_generation::ImageGenerationResponse<GenerateContentResponse>
{
    type Error = ImageGenerationError;

    fn try_from(value: GenerateContentResponse) -> Result<Self, Self::Error> {
        let image = first_image_bytes(&value)?;

        Ok(image_generation::ImageGenerationResponse {
            image,
            response: value,
        })
    }
}

impl<T> image_generation::ImageGenerationModel for ImageGenerationModel<T>
where
    T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
{
    type Response = GenerateContentResponse;

    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model)
    }

    async fn image_generation(
        &self,
        generation_request: ImageGenerationRequest,
    ) -> Result<image_generation::ImageGenerationResponse<Self::Response>, ImageGenerationError>
    {
        let body = serde_json::to_vec(&create_request_body(generation_request)?)?;

        let request = self
            .client
            .post(generate_content_path(&self.model))?
            .body(body)
            .map_err(|e| ImageGenerationError::HttpError(e.into()))?;

        let response = self.client.send(request).await?;

        let status = response.status();
        let text = http_client::text(response).await?;

        if !status.is_success() {
            return Err(ImageGenerationError::from_http_response(status, text));
        }

        match serde_json::from_str::<ApiResponse<GenerateContentResponse>>(&text)? {
            ApiResponse::Ok(response) => response.try_into(),
            // Gemini returns its error envelope with a 2xx status; preserve the
            // raw body alongside that status instead of flattening the message.
            ApiResponse::Err(err) => {
                tracing::warn!(message = %err.message, "provider returned an error response");
                Err(ImageGenerationError::from_http_response(status, text))
            }
        }
    }
}

fn generate_content_path(model: &str) -> String {
    format!("/v1beta/models/{model}:generateContent")
}

fn create_request_body(
    generation_request: ImageGenerationRequest,
) -> Result<Value, ImageGenerationError> {
    let request = GenerateContentRequest {
        contents: vec![Content {
            role: Some(Role::User),
            parts: vec![Part {
                thought: None,
                thought_signature: None,
                part: PartKind::Text(generation_request.prompt),
                additional_params: None,
            }],
        }],
        tools: None,
        tool_config: None,
        generation_config: Some(GenerationConfig {
            response_modalities: Some(vec![ResponseModality::Image]),
            image_config: Some(ImageConfig {
                aspect_ratio: aspect_ratio(generation_request.width, generation_request.height),
                image_size: None,
            }),
            ..Default::default()
        }),
        safety_settings: None,
        system_instruction: None,
        additional_params: None,
    };

    let mut body = serde_json::to_value(request)?;

    if let Some(additional_params) = generation_request.additional_params {
        merge_json_deep(&mut body, additional_params);
    }

    Ok(body)
}

fn merge_json_deep(target: &mut Value, source: Value) {
    match (target, source) {
        (Value::Object(target), Value::Object(source)) => {
            for (key, value) in source {
                if let Some(existing) = target.get_mut(&key) {
                    merge_json_deep(existing, value);
                } else {
                    target.insert(key, value);
                }
            }
        }
        (target, source) => *target = source,
    }
}

fn aspect_ratio(width: u32, height: u32) -> Option<String> {
    match (width, height) {
        (0, _) | (_, 0) => None,
        (w, h) if w == h => Some("1:1".to_string()),
        (w, h) if w.saturating_mul(3) == h.saturating_mul(4) => Some("3:4".to_string()),
        (w, h) if w.saturating_mul(4) == h.saturating_mul(3) => Some("4:3".to_string()),
        (w, h) if w.saturating_mul(9) == h.saturating_mul(16) => Some("9:16".to_string()),
        (w, h) if w.saturating_mul(16) == h.saturating_mul(9) => Some("16:9".to_string()),
        _ => None,
    }
}

fn first_image_bytes(response: &GenerateContentResponse) -> Result<Vec<u8>, ImageGenerationError> {
    for candidate in &response.candidates {
        let Some(content) = &candidate.content else {
            continue;
        };

        for part in &content.parts {
            if part.thought == Some(true) {
                continue;
            }

            if let PartKind::InlineData(inline_data) = &part.part {
                if !inline_data.mime_type.starts_with("image/") {
                    continue;
                }

                return BASE64_STANDARD.decode(&inline_data.data).map_err(|err| {
                    ImageGenerationError::ResponseError(format!(
                        "Gemini image data was not valid base64: {err}"
                    ))
                });
            }
        }
    }

    Err(ImageGenerationError::ResponseError(
        "Gemini image generation response did not include image data".into(),
    ))
}

#[cfg(test)]
mod tests {
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

        let parsed: image_generation::ImageGenerationResponse<GenerateContentResponse> = response
            .try_into()
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

        let err = image_generation::ImageGenerationResponse::<GenerateContentResponse>::try_from(
            response,
        )
        .expect_err("text-only responses should fail");

        assert!(err.to_string().contains("did not include image data"));
    }

    #[tokio::test]
    async fn image_generation_non_success_response_preserves_status_and_body() {
        use crate::image_generation::ImageGenerationModel as _;
        use crate::test_utils::RecordingHttpClient;

        let body =
            r#"{"error":{"code":400,"message":"invalid request","status":"INVALID_ARGUMENT"}}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::BAD_REQUEST, body);
        let client = Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = ImageGenerationModel::new(client, GEMINI_2_5_FLASH_IMAGE);

        let error = model
            .image_generation(image_generation_request("draw a cat"))
            .await
            .expect_err("image generation should fail with non-success status");

        assert!(matches!(error, ImageGenerationError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::BAD_REQUEST)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }

    #[tokio::test]
    async fn image_generation_preserves_provider_error_envelope_on_2xx() {
        use crate::image_generation::ImageGenerationModel as _;
        use crate::test_utils::RecordingHttpClient;

        // Gemini returns its error envelope with a 2xx status; the raw body and
        // status should be preserved as a `ProviderResponse`.
        let body = r#"{"message":"image generation blocked by safety filters"}"#;
        let http_client = RecordingHttpClient::new(body);
        let client = Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = ImageGenerationModel::new(client, GEMINI_2_5_FLASH_IMAGE);

        let error = model
            .image_generation(image_generation_request("draw a cat"))
            .await
            .expect_err("image generation should fail with provider error envelope");

        match &error {
            ImageGenerationError::ProviderResponse(stored) => {
                assert_eq!(stored.body, body);
                assert_eq!(stored.status, Some(http::StatusCode::OK));
                assert_eq!(error.provider_response_body(), Some(body));
                assert_eq!(error.provider_response_status(), Some(http::StatusCode::OK));
            }
            other => panic!("expected ProviderResponse, got {other:?}"),
        }
    }
}
