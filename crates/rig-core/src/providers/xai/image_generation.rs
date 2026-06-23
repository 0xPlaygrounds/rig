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
        let mut request = json!({
            "model": self.model,
            "prompt": generation_request.prompt,
            "response_format": "b64_json",
            "aspect_ratio": "1:1",
        });

        if let Some(additional_params) = generation_request.additional_params {
            merge_inplace(&mut request, additional_params);
        }

        let body = serde_json::to_vec(&request)?;

        let request = self
            .client
            .post("/v1/images/generations")?
            .body(body)
            .map_err(|e| ImageGenerationError::HttpError(e.into()))?;

        let response = self.client.send(request).await?;

        let status = response.status();
        let text = http_client::text(response).await?;

        if !status.is_success() {
            return Err(ImageGenerationError::from_http_response(status, text));
        }

        match serde_json::from_str::<ApiResponse<ImageGenerationResponse>>(&text)? {
            ApiResponse::Ok(response) => response.try_into(),
            // xAI returns its error envelope with a 2xx status; preserve the raw
            // body alongside that status instead of flattening the message.
            ApiResponse::Error(err) => {
                tracing::warn!(message = %err.message(), "provider returned an error response");
                Err(ImageGenerationError::from_http_response(status, text))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::client::image_generation::ImageGenerationClient;
    use crate::image_generation::{ImageGenerationError, ImageGenerationModel as _};
    use crate::test_utils::RecordingHttpClient;

    #[tokio::test]
    async fn image_generation_non_success_response_preserves_status_and_body() {
        let body = r#"{"error":"invalid prompt","code":"400"}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::BAD_REQUEST, body);
        let client = crate::providers::xai::Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.image_generation_model(super::GROK_IMAGINE_IMAGE);

        let error = model
            .image_generation_request()
            .prompt("draw a cat")
            .width(256)
            .height(256)
            .send()
            .await
            .err()
            .expect("image generation should fail with non-success status");

        assert!(matches!(error, ImageGenerationError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::BAD_REQUEST)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }
}
