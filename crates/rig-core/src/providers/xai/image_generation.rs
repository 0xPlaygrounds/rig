use super::api::ApiResponse;
use super::client::Client;
use crate::http_client::HttpClientExt;
use crate::image_generation::{ImageGenerationError, ImageGenerationRequest};
use crate::json_utils::merge_inplace;
use crate::{http_client, image_generation};
use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use serde::Deserialize;
use serde_json::json;

// ================================================================
// xAI Image Generation API
// ================================================================
pub const GROK_IMAGINE_IMAGE: &str = "grok-imagine-image";
pub const GROK_IMAGINE_IMAGE_PRO: &str = "grok-imagine-image-pro";

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

#[derive(Clone)]
pub struct ImageGenerationModel<T = reqwest::Client> {
    client: Client<T>,
    /// Name of the model (e.g.: grok-imagine-image)
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

impl<T> image_generation::ImageGenerationModel for ImageGenerationModel<T>
where
    T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
{
    type Response = ImageGenerationResponse;

    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model)
    }

    async fn image_generation(
        &self,
        generation_request: ImageGenerationRequest,
    ) -> Result<image_generation::ImageGenerationResponse<Self::Response>, ImageGenerationError>
    {
        let body =
            build_image_generation_body(&self.model, &generation_request)?;

        let request = self
            .client
            .post("/v1/images/generations")?
            .body(body)
            .map_err(|e| ImageGenerationError::HttpError(e.into()))?;

        let response = self.client.send(request).await?;

        let status = response.status();
        let text = http_client::text(response).await?;
        parse_image_generation_response(status, &text)
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
        "aspect_ratio": "1:1",
    });

    if let Some(additional_params) = request.additional_params.clone() {
        merge_inplace(&mut body, additional_params);
    }

    Ok(serde_json::to_vec(&body)?)
}

/// Parse an image-generation response body. Pure.
pub(crate) fn parse_image_generation_response(
    status: http::StatusCode,
    text: &str,
) -> Result<
    image_generation::ImageGenerationResponse<ImageGenerationResponse>,
    ImageGenerationError,
> {
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
    use crate::client::image_generation::ImageGenerationClient;
    use crate::image_generation::ImageGenerationModel as _;

    fn request() -> ImageGenerationRequest {
        ImageGenerationRequest {
            prompt: "draw a cat".to_string(),
            width: 256,
            height: 256,
            additional_params: None,
        }
    }

    #[test]
    fn image_generation_body_merges_additional_params() {
        let mut generation_request = request();
        generation_request.additional_params = Some(serde_json::json!({
            "aspect_ratio": "16:9",
            "n": 2,
        }));
        let body =
            build_image_generation_body(GROK_IMAGINE_IMAGE, &generation_request).expect("build");
        let value: serde_json::Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(value["model"], GROK_IMAGINE_IMAGE);
        assert_eq!(value["prompt"], "draw a cat");
        assert_eq!(value["response_format"], "b64_json");
        assert_eq!(value["aspect_ratio"], "16:9");
        assert_eq!(value["n"], 2);
    }

    #[tokio::test]
    async fn image_generation_non_success_preserves_status_and_body() {
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"error":"boom","code":"503"}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
        let client = crate::providers::xai::Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.image_generation_model(GROK_IMAGINE_IMAGE);

        let error = model
            .image_generation(request())
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
        let http_client = RecordingHttpClient::new(body);
        let client = crate::providers::xai::Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.image_generation_model(GROK_IMAGINE_IMAGE);

        let error = model
            .image_generation(request())
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
