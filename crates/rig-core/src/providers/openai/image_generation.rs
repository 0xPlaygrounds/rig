use super::{Client, client::ApiResponse};
use crate::http_client::HttpClientExt;
use crate::image_generation::{ImageGenerationError, ImageGenerationRequest};
use crate::json_utils::merge_inplace;
use crate::{http_client, image_generation};
use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use serde::Deserialize;
use serde_json::json;

// ================================================================
// OpenAI Image Generation API
// ================================================================
pub const DALL_E_2: &str = "dall-e-2";
pub const DALL_E_3: &str = "dall-e-3";
pub const GPT_IMAGE_1: &str = "gpt-image-1";
pub const GPT_IMAGE_1_5: &str = "gpt-image-1.5";
pub const GPT_IMAGE_2: &str = "gpt-image-2";

#[derive(Debug, Deserialize)]
pub struct ImageGenerationData {
    pub b64_json: String,
}

#[derive(Debug, Deserialize)]
pub struct ImageGenerationResponse {
    pub created: i32,
    pub data: Vec<ImageGenerationData>,
}

impl TryFrom<ImageGenerationResponse>
    for image_generation::ImageGenerationResponse<ImageGenerationResponse>
{
    type Error = ImageGenerationError;

    fn try_from(value: ImageGenerationResponse) -> Result<Self, Self::Error> {
        let b64_json = value
            .data
            .first()
            .ok_or_else(|| ImageGenerationError::ResponseError("missing image data".into()))?
            .b64_json
            .clone();

        let bytes = BASE64_STANDARD
            .decode(&b64_json)
            .map_err(|err| ImageGenerationError::ResponseError(err.to_string()))?;

        Ok(image_generation::ImageGenerationResponse {
            image: bytes,
            response: value,
        })
    }
}

#[derive(Clone)]
pub struct ImageGenerationModel<T = reqwest::Client> {
    client: Client<T>,
    /// Name of the model (e.g.: dall-e-2)
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
        let mut request = json!({
            "model": self.model,
            "prompt": generation_request.prompt,
            "size": format!("{}x{}", generation_request.width, generation_request.height),
        });

        if !matches!(
            self.model.as_str(),
            GPT_IMAGE_1 | GPT_IMAGE_1_5 | GPT_IMAGE_2
        ) {
            merge_inplace(
                &mut request,
                json!({
                    "response_format": "b64_json"
                }),
            );
        }

        let body = serde_json::to_vec(&request)?;

        let request = self
            .client
            .post("/images/generations")?
            .body(body)
            .map_err(|e| ImageGenerationError::HttpError(e.into()))?;

        let response = self.client.send(request).await?;

        let status = response.status();
        if !status.is_success() {
            let text = http_client::text(response).await?;

            return Err(ImageGenerationError::HttpError(
                http_client::Error::InvalidStatusCodeWithMessage(status, text),
            ));
        }

        let text = http_client::text(response).await?;

        match serde_json::from_str::<ApiResponse<ImageGenerationResponse>>(&text)? {
            ApiResponse::Ok(response) => response.try_into(),
            ApiResponse::Err(err) => {
                tracing::warn!(message = %err.message, "provider returned an error response");
                Err(ImageGenerationError::ProviderResponse(
                    crate::provider_response::ProviderResponseError {
                        status: Some(status),
                        body: text,
                    },
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::image_generation::ImageGenerationClient;
    use crate::image_generation::ImageGenerationModel as _;
    use crate::test_utils::RecordingHttpClient;

    fn request() -> ImageGenerationRequest {
        ImageGenerationRequest {
            prompt: "draw a cat".to_string(),
            width: 256,
            height: 256,
            additional_params: None,
        }
    }

    #[tokio::test]
    async fn image_generation_non_success_response_preserves_status_and_body() {
        let body = r#"{"error":{"message":"invalid image","type":"invalid_request_error"}}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::BAD_REQUEST, body);
        let client = Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.image_generation_model(DALL_E_3);

        let error = model
            .image_generation(request())
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
    async fn image_generation_preserves_raw_provider_error_json_on_api_error_envelope() {
        let body = r#"{"message":"quota exceeded","type":"insufficient_quota"}"#;
        let http_client = RecordingHttpClient::new(body);
        let client = Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.image_generation_model(DALL_E_3);

        let error = model
            .image_generation(request())
            .await
            .expect_err("image generation should fail with provider error envelope");

        match &error {
            ImageGenerationError::ProviderResponse(stored) => {
                assert_eq!(stored.body, body);
                assert_eq!(stored.status, Some(http::StatusCode::OK));
                assert_eq!(error.provider_response_body(), Some(body));
            }
            other => panic!("expected ProviderResponse, got {other:?}"),
        }
    }
}
