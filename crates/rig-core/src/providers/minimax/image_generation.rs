//! MiniMax image generation support.

use super::Client;
use crate::http_client::HttpClientExt;
use crate::image_generation::{ImageGenerationError, ImageGenerationRequest};
use crate::json_utils::merge_inplace;
use crate::{http_client, image_generation};
use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use bytes::Bytes;
use serde::Deserialize;
use serde_json::json;

/// `image-01` image generation model.
pub const IMAGE_01: &str = "image-01";
/// `image-01-live` image generation model.
pub const IMAGE_01_LIVE: &str = "image-01-live";

/// Image data returned by MiniMax.
#[derive(Debug, Default, Deserialize)]
pub struct ImageGenerationData {
    /// Generated image URLs when URL output is requested.
    #[serde(default)]
    pub image_urls: Vec<String>,
    /// Base64-encoded images when base64 output is requested.
    #[serde(default)]
    pub image_base64: Vec<String>,
}

/// MiniMax image generation status information.
#[derive(Debug, Default, Deserialize)]
pub struct BaseResponse {
    /// Provider status code, where zero indicates success.
    #[serde(default)]
    pub status_code: i64,
    /// Provider status message.
    #[serde(default)]
    pub status_msg: String,
}

/// Raw MiniMax image generation response.
#[derive(Debug, Deserialize)]
pub struct ImageGenerationResponse {
    /// Generated image data.
    #[serde(default)]
    pub data: ImageGenerationData,
    /// Generation metadata such as success and failure counts.
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
    /// Request trace identifier.
    #[serde(default)]
    pub id: Option<String>,
    /// Provider status information.
    #[serde(default)]
    pub base_resp: BaseResponse,
}

/// MiniMax image generation model.
#[derive(Clone)]
pub struct ImageGenerationModel<T = reqwest::Client> {
    client: Client<T>,
    /// Name of the image generation model.
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
        });

        if valid_dimensions(generation_request.width, generation_request.height) {
            merge_inplace(
                &mut request,
                json!({
                    "width": generation_request.width,
                    "height": generation_request.height,
                }),
            );
        }

        if let Some(additional_params) = generation_request.additional_params {
            merge_inplace(&mut request, additional_params);
        }

        let body = serde_json::to_vec(&request)?;
        let request = self
            .client
            .post("/image_generation")?
            .body(body)
            .map_err(|err| ImageGenerationError::HttpError(err.into()))?;

        let response = self.client.send(request).await?;
        let status = response.status();
        let text = http_client::text(response).await?;

        if !status.is_success() {
            return Err(ImageGenerationError::from_http_response(status, text));
        }

        let response = match serde_json::from_str::<ImageGenerationResponse>(&text) {
            Ok(response) => response,
            Err(error) => {
                tracing::warn!(message = %error, "provider returned an invalid response");
                return Err(ImageGenerationError::from_http_response(status, text));
            }
        };

        if response.base_resp.status_code != 0
            || (response.data.image_base64.is_empty() && response.data.image_urls.is_empty())
        {
            return Err(ImageGenerationError::from_http_response(status, text));
        }

        let image = self.first_image_bytes(&response).await?;

        Ok(image_generation::ImageGenerationResponse { image, response })
    }
}

impl<T> ImageGenerationModel<T>
where
    T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
{
    async fn first_image_bytes(
        &self,
        response: &ImageGenerationResponse,
    ) -> Result<Vec<u8>, ImageGenerationError> {
        if let Some(image) = response.data.image_base64.first() {
            return decode_base64_image(image);
        }

        let url = response
            .data
            .image_urls
            .first()
            .ok_or_else(|| ImageGenerationError::ResponseError("missing image data".into()))?;

        if url.starts_with("data:") {
            return decode_base64_image(url);
        }

        self.download_image(url).await
    }

    async fn download_image(&self, url: &str) -> Result<Vec<u8>, ImageGenerationError> {
        let request = http_client::Request::get(url)
            .body(http_client::NoBody)
            .map_err(http_client::Error::from)?;
        let response = self.client.send::<_, Bytes>(request).await?;
        let status = response.status();
        let body = response.into_body().await?.to_vec();

        if !status.is_success() {
            return Err(ImageGenerationError::from_http_response(
                status,
                String::from_utf8_lossy(&body),
            ));
        }

        Ok(body)
    }
}

fn valid_dimensions(width: u32, height: u32) -> bool {
    [width, height]
        .into_iter()
        .all(|dimension| (512..=2048).contains(&dimension) && dimension % 8 == 0)
}

fn decode_base64_image(value: &str) -> Result<Vec<u8>, ImageGenerationError> {
    let payload = value
        .split_once(',')
        .filter(|(metadata, _)| metadata.starts_with("data:") && metadata.ends_with(";base64"))
        .map_or(value, |(_, payload)| payload);

    BASE64_STANDARD.decode(payload).map_err(|error| {
        ImageGenerationError::ResponseError(format!("invalid base64 image data: {error}"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::image_generation::ImageGenerationClient;
    use crate::image_generation::ImageGenerationModel as _;
    use crate::test_utils::{MockHttpResponse, RecordingHttpClient, SequencedHttpClient};

    fn request(additional_params: Option<serde_json::Value>) -> ImageGenerationRequest {
        ImageGenerationRequest {
            prompt: "draw a mountain lake".to_string(),
            width: 1024,
            height: 768,
            additional_params,
        }
    }

    fn success_body() -> &'static str {
        r#"{"data":{"image_base64":["aW1hZ2U="]},"metadata":{"success_count":1,"failed_count":0},"id":"trace-id","base_resp":{"status_code":0,"status_msg":"success"}}"#
    }

    #[tokio::test]
    async fn base64_response_maps_supported_request_fields() {
        let http_client = RecordingHttpClient::new(success_body());
        let recorded = http_client.clone();
        let client = Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.image_generation_model(IMAGE_01);

        let response = model
            .image_generation(request(Some(json!({
                "subject_reference": [{"type": "character", "image_file": "reference-id"}],
                "aspect_ratio": "4:3",
                "response_format": "base64",
                "seed": 7,
                "n": 2,
                "prompt_optimizer": true,
            }))))
            .await
            .expect("generate image");

        assert_eq!(response.image, b"image");
        let requests = recorded.requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(
            requests[0].uri,
            "https://api.minimax.io/v1/image_generation"
        );

        let body: serde_json::Value =
            serde_json::from_slice(&requests[0].body).expect("request JSON");
        assert_eq!(body["model"], IMAGE_01);
        assert_eq!(body["prompt"], "draw a mountain lake");
        assert_eq!(body["width"], 1024);
        assert_eq!(body["height"], 768);
        assert_eq!(body["subject_reference"][0]["image_file"], "reference-id");
        assert_eq!(body["aspect_ratio"], "4:3");
        assert_eq!(body["response_format"], "base64");
        assert_eq!(body["seed"], 7);
        assert_eq!(body["n"], 2);
        assert_eq!(body["prompt_optimizer"], true);
    }

    #[tokio::test]
    async fn china_client_uses_china_image_endpoint_and_omits_default_dimensions() {
        let http_client = RecordingHttpClient::new(success_body());
        let recorded = http_client.clone();
        let client = Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .china()
            .build()
            .expect("build client");
        let model = client.image_generation_model(IMAGE_01_LIVE);

        model
            .image_generation(ImageGenerationRequest {
                prompt: "draw a mountain lake".to_string(),
                width: 256,
                height: 256,
                additional_params: Some(json!({"response_format": "base64"})),
            })
            .await
            .expect("generate image");

        let requests = recorded.requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(
            requests[0].uri,
            "https://api.minimaxi.com/v1/image_generation"
        );
        let body: serde_json::Value =
            serde_json::from_slice(&requests[0].body).expect("request JSON");
        assert!(body.get("width").is_none());
        assert!(body.get("height").is_none());
    }

    #[tokio::test]
    async fn url_response_downloads_without_forwarding_authorization() {
        let image_url = "https://cdn.example.com/generated.png";
        let response_body = format!(
            r#"{{"data":{{"image_urls":["{image_url}"]}},"base_resp":{{"status_code":0,"status_msg":"success"}}}}"#
        );
        let http_client = SequencedHttpClient::new([
            MockHttpResponse::success(response_body),
            MockHttpResponse::success(Bytes::from_static(b"downloaded-image")),
        ]);
        let recorded = http_client.clone();
        let client = Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.image_generation_model(IMAGE_01);

        let response = model
            .image_generation(request(None))
            .await
            .expect("generate image");

        assert_eq!(response.image, b"downloaded-image");
        let requests = recorded.requests();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[1].uri, image_url);
        assert!(
            !requests[1]
                .headers
                .contains_key(http::header::AUTHORIZATION)
        );
    }

    #[tokio::test]
    async fn nonzero_provider_status_preserves_success_response_body() {
        let body =
            r#"{"data":{},"base_resp":{"status_code":1008,"status_msg":"insufficient balance"}}"#;
        let http_client = RecordingHttpClient::new(body);
        let client = Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.image_generation_model(IMAGE_01);

        let error = model
            .image_generation(request(None))
            .await
            .expect_err("provider status should fail");

        assert!(matches!(error, ImageGenerationError::ProviderResponse(_)));
        assert_eq!(error.provider_response_status(), Some(http::StatusCode::OK));
        assert_eq!(error.provider_response_body(), Some(body));
    }

    #[tokio::test]
    async fn http_failure_preserves_status_and_body() {
        let body = r#"{"base_resp":{"status_code":2013,"status_msg":"invalid parameters"}}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::BAD_REQUEST, body);
        let client = Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.image_generation_model(IMAGE_01);

        let error = model
            .image_generation(request(None))
            .await
            .expect_err("HTTP status should fail");

        assert!(matches!(error, ImageGenerationError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::BAD_REQUEST)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }

    #[test]
    fn data_url_response_decodes_inline_image() {
        assert_eq!(
            decode_base64_image("data:image/png;base64,aW1hZ2U=").expect("decode image"),
            b"image"
        );
    }
}
