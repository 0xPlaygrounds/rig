use super::api::ApiResponse;
use crate::image_generation;
use crate::image_generation::{ImageGenerationError, ImageGenerationRequest};
use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use serde::Deserialize;
use serde_json::json;

// ================================================================
// xAI Image Generation API
// ================================================================
pub const GROK_IMAGINE_IMAGE: &str = "grok-imagine-image";
pub const GROK_IMAGINE_IMAGE_PRO: &str = "grok-imagine-image-pro";

#[derive(Debug, thiserror::Error)]
#[error(
    "xAI image generation cannot represent dimensions {width}x{height}; use a supported ratio or set `additional_params.aspect_ratio` explicitly"
)]
struct UnsupportedImageDimensions {
    width: u32,
    height: u32,
}

#[derive(Debug, Deserialize)]
pub struct ImageGenerationData {
    pub b64_json: String,
}

#[derive(Debug, Deserialize)]
pub struct ImageGenerationResponse {
    pub data: Vec<ImageGenerationData>,
}

impl TryFrom<ImageGenerationResponse>
    for image_generation::ImageGenerationResponse<ImageGenerationResponse>
{
    type Error = ImageGenerationError;

    fn try_from(value: ImageGenerationResponse) -> Result<Self, Self::Error> {
        let first = value
            .data
            .first()
            .ok_or_else(|| ImageGenerationError::ResponseError("No image data returned".into()))?;

        let bytes = BASE64_STANDARD.decode(&first.b64_json).map_err(|e| {
            ImageGenerationError::ResponseError(format!("Base64 decode error: {e}"))
        })?;

        Ok(image_generation::ImageGenerationResponse {
            image: bytes,
            response: value,
        })
    }
}

/// Build the serialized image-generation request body. Pure.
pub(crate) fn build_image_generation_body(
    model: &str,
    request: &ImageGenerationRequest,
) -> Result<Vec<u8>, ImageGenerationError> {
    let mut body = json!({
        "model": model,
        "prompt": request.prompt,
        "response_format": "b64_json",
    });
    body = crate::json_utils::merge_additional_params(
        body,
        request.additional_params.clone(),
        &["model", "prompt", "response_format"],
        "xAI image-generation request",
    )?;
    if body.get("aspect_ratio").is_none() {
        let aspect_ratio = aspect_ratio(request.width, request.height).ok_or_else(|| {
            ImageGenerationError::RequestError(Box::new(UnsupportedImageDimensions {
                width: request.width,
                height: request.height,
            }))
        })?;
        body.as_object_mut()
            .ok_or(crate::json_utils::RequestOverlayError::CanonicalNotObject {
                context: "xAI image-generation request",
            })?
            .insert("aspect_ratio".to_string(), json!(aspect_ratio));
    }

    Ok(serde_json::to_vec(&body)?)
}

fn aspect_ratio(width: u32, height: u32) -> Option<&'static str> {
    const RATIOS: &[(&str, u64, u64)] = &[
        ("1:1", 1, 1),
        ("16:9", 16, 9),
        ("9:16", 9, 16),
        ("4:3", 4, 3),
        ("3:4", 3, 4),
        ("3:2", 3, 2),
        ("2:3", 2, 3),
        ("2:1", 2, 1),
        ("1:2", 1, 2),
        ("19.5:9", 39, 18),
        ("9:19.5", 18, 39),
        ("20:9", 20, 9),
        ("9:20", 9, 20),
    ];

    let width = u64::from(width);
    let height = u64::from(height);
    if width == 0 || height == 0 {
        return None;
    }

    RATIOS
        .iter()
        .find(|(_, ratio_width, ratio_height)| width * ratio_height == height * ratio_width)
        .map(|(name, _, _)| *name)
}

/// Parse an image-generation response body. Pure.
pub(crate) fn parse_image_generation_response(
    status: http::StatusCode,
    text: &str,
) -> Result<image_generation::ImageGenerationResponse<ImageGenerationResponse>, ImageGenerationError>
{
    if !status.is_success() {
        return Err(ImageGenerationError::from_http_response(
            status,
            text.to_string(),
        ));
    }

    match serde_json::from_str::<ApiResponse<ImageGenerationResponse>>(text)? {
        ApiResponse::Ok(response) => response.try_into(),
        ApiResponse::Error(err) => {
            tracing::warn!(message = %err.message(), "provider returned an error response");
            Err(ImageGenerationError::from_http_response(
                status,
                text.to_string(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::xai::functions;

    fn request() -> ImageGenerationRequest {
        ImageGenerationRequest {
            prompt: "draw a cat".to_string(),
            width: 256,
            height: 256,
            additional_params: None,
        }
    }

    #[test]
    fn image_generation_body_prefers_explicit_provider_aspect_ratio() {
        let mut generation_request = request();
        generation_request.additional_params = Some(serde_json::json!({
            "aspect_ratio": "16:9",
            "resolution": "2k",
        }));
        let body = build_image_generation_body(GROK_IMAGINE_IMAGE, &generation_request)
            .expect("an explicit provider-native ratio should override generic dimensions");
        let value: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(value["aspect_ratio"], "16:9");
        assert_eq!(value["resolution"], "2k");
    }

    #[test]
    fn image_generation_maps_generic_dimensions_to_provider_aspect_ratio() {
        let mut generation_request = request();
        generation_request.width = 1920;
        generation_request.height = 1080;

        let body = build_image_generation_body(GROK_IMAGINE_IMAGE, &generation_request)
            .expect("supported dimensions should map to an aspect ratio");
        let value: serde_json::Value = serde_json::from_slice(&body).expect("json");

        assert_eq!(value["aspect_ratio"], "16:9");
    }

    #[test]
    fn image_generation_requires_explicit_ratio_for_unsupported_dimensions() {
        let mut generation_request = request();
        generation_request.width = 5;
        generation_request.height = 3;

        let error = build_image_generation_body(GROK_IMAGINE_IMAGE, &generation_request)
            .expect_err("unsupported dimensions must not be silently ignored");
        assert!(matches!(error, ImageGenerationError::RequestError(_)));
        assert!(error.to_string().contains("5x3"));

        generation_request.additional_params = Some(serde_json::json!({"aspect_ratio": "auto"}));
        let body = build_image_generation_body(GROK_IMAGINE_IMAGE, &generation_request)
            .expect("an explicit provider-native ratio should be accepted");
        let value: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(value["aspect_ratio"], "auto");
    }

    #[test]
    fn image_generation_body_preserves_unrelated_additional_params() {
        let mut generation_request = request();
        generation_request.additional_params = Some(serde_json::json!({"n": 2}));
        let body = build_image_generation_body(GROK_IMAGINE_IMAGE, &generation_request)
            .expect("unrelated extension should be accepted");
        let value: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(value["model"], GROK_IMAGINE_IMAGE);
        assert_eq!(value["prompt"], "draw a cat");
        assert_eq!(value["response_format"], "b64_json");
        assert_eq!(value["aspect_ratio"], "1:1");
        assert_eq!(value["n"], 2);
    }

    #[tokio::test]
    async fn image_generation_non_success_preserves_status_and_body() {
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"error":"boom","code":"503"}"#;
        let rt = crate::http_runtime::HttpRuntime::recording(
            RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body),
        );
        let cfg = functions::Config::new(GROK_IMAGINE_IMAGE).with_api_key("test-key");

        let error = functions::generate_image(&cfg, &rt, request())
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
        use crate::test_utils::RecordingHttpClient;

        // Deserializes to `ApiResponse::Error(ApiError { error, code })` on a 200 OK.
        let body = r#"{"error":"boom","code":"503"}"#;
        let rt = crate::http_runtime::HttpRuntime::recording(RecordingHttpClient::new(body));
        let cfg = functions::Config::new(GROK_IMAGINE_IMAGE).with_api_key("test-key");

        let error = functions::generate_image(&cfg, &rt, request())
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
}
