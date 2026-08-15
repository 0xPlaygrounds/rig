//! Shared request driver for OpenAI-style image generation endpoints.
//!
//! OpenAI, Azure OpenAI, xAI and Hyperbolic each build a provider-specific
//! JSON body, but the tail is identical: POST the body, classify the response
//! through the provider's success-or-error envelope, and convert the payload.
//! The body and path stay with each provider — those are the real wire
//! differences — while this driver owns the shared send/decode tail.

use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use bytes::Bytes;
use serde::de::DeserializeOwned;

use super::envelope::ProviderEnvelope;
use crate::client::{Client, Provider};
use crate::http_client::{self, HttpClientExt};
use crate::image_generation::{self, ImageGenerationError, ImageGenerationRequest};
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};

/// Decodes the first base64 image selected from a provider response while
/// retaining that response in Rig's normalized wrapper.
pub(crate) fn decode_base64_image<T>(
    response: T,
    select: fn(&T) -> Option<&str>,
    missing_message: &'static str,
    decode_error_prefix: Option<&'static str>,
) -> Result<image_generation::ImageGenerationResponse<T>, ImageGenerationError> {
    let encoded = select(&response)
        .ok_or_else(|| ImageGenerationError::ResponseError(missing_message.to_owned()))?;
    let image = BASE64_STANDARD.decode(encoded).map_err(|error| {
        ImageGenerationError::ResponseError(match decode_error_prefix {
            Some(prefix) => format!("{prefix}{error}"),
            None => error.to_string(),
        })
    })?;
    Ok(image_generation::ImageGenerationResponse { image, response })
}

/// Provider-specific request and response types for the shared OpenAI-envelope
/// image generation model.
#[doc(hidden)]
pub trait JsonImageGenerationProvider: Provider {
    const IMAGE_GENERATION_PATH: &'static str;

    type Response: DeserializeOwned
        + WasmCompatSend
        + WasmCompatSync
        + TryInto<
            image_generation::ImageGenerationResponse<Self::Response>,
            Error = ImageGenerationError,
        >;
    fn image_generation_request_builder<H>(
        client: &Client<Self, H>,
        _model: &str,
    ) -> Result<http_client::Builder, ImageGenerationError>
    where
        H: HttpClientExt,
    {
        Ok(client.post(Self::IMAGE_GENERATION_PATH)?)
    }

    fn image_generation_request_body(
        model: &str,
        request: ImageGenerationRequest,
    ) -> Result<serde_json::Value, ImageGenerationError>;
}

/// Shared model shell for JSON image-generation endpoints.
#[doc(hidden)]
#[derive(Clone)]
pub struct GenericImageGenerationModel<Ext, H = reqwest::Client> {
    client: Client<Ext, H>,
    /// Name of the image generation model.
    pub model: String,
}

impl<Ext, H> GenericImageGenerationModel<Ext, H> {
    /// Creates an image generation model backed by `client`.
    pub fn new(client: Client<Ext, H>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }

    /// Creates an image generation model from a borrowed model name.
    pub fn with_model(client: Client<Ext, H>, model: &str) -> Self {
        Self::new(client, model)
    }
}

impl<Ext, H> image_generation::ImageGenerationModel for GenericImageGenerationModel<Ext, H>
where
    Ext: JsonImageGenerationProvider + Clone + WasmCompatSend + WasmCompatSync + 'static,
    H: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    type Response = Ext::Response;
    type Client = Client<Ext, H>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model)
    }

    async fn image_generation(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<image_generation::ImageGenerationResponse<Self::Response>, ImageGenerationError>
    {
        let builder = Ext::image_generation_request_builder(&self.client, &self.model)?;
        let body = Ext::image_generation_request_body(&self.model, request)?;
        send_image_generation::<_, crate::providers::openai::client::ApiResponse<Ext::Response>>(
            &self.client,
            builder,
            body,
        )
        .await
    }
}

/// Sends an image generation request and decodes the shared success-or-error
/// envelope.
///
/// `builder` is the provider's already-path-built POST request; `body` is the
/// provider's JSON request body; `A` is the provider's own response envelope
/// so error-body classification is unchanged. Provider error bodies are
/// preserved raw via [`ImageGenerationError::from_http_response`].
pub(crate) async fn send_image_generation<C, A>(
    client: &C,
    builder: http_client::Builder,
    body: serde_json::Value,
) -> Result<image_generation::ImageGenerationResponse<A::Payload>, ImageGenerationError>
where
    C: HttpClientExt,
    A: DeserializeOwned + ProviderEnvelope,
    A::Payload: TryInto<image_generation::ImageGenerationResponse<A::Payload>, Error = ImageGenerationError>,
{
    let body = serde_json::to_vec(&body)?;

    let req = builder
        .body(body)
        .map_err(|e| ImageGenerationError::HttpError(e.into()))?;

    let response = client.send::<_, Bytes>(req).await?;

    // Taking the response apart hands the headers over already owned, so both
    // failure paths keep their rate-limit metadata at no cost to the success
    // path (rig#2210).
    let (parts, body) = response.into_parts();
    let status = parts.status;
    let headers = Box::new(parts.headers);
    let response_body = body.into_future().await?;

    if !status.is_success() {
        return Err(ImageGenerationError::from_http_response(
            status,
            String::from_utf8_lossy(&response_body).into_owned(),
        )
        .with_response_headers(Some(headers)));
    }

    match serde_json::from_slice::<A>(&response_body)?.into_payload() {
        Ok(response) => response.try_into(),
        Err(message) => {
            tracing::warn!(message = %message, "provider returned an error response");
            Err(ImageGenerationError::from_http_response(
                status,
                String::from_utf8_lossy(&response_body).into_owned(),
            )
            .with_response_headers(Some(headers)))
        }
    }
}

/// rig#2210: a failed image-generation response keeps its headers, so the
/// capability error's `provider_response_headers()` is not a promise the
/// driver quietly breaks.
#[cfg(test)]
mod header_preservation_tests {
    use super::*;
    use crate::providers::internal::envelope::DirectPayload;
    use crate::test_utils::RecordingHttpClient;

    /// Minimal payload satisfying the driver's `TryInto` bound; the 429 path
    /// returns before any decoding, so its conversion is never reached.
    #[derive(serde::Deserialize)]
    struct Payload;

    impl TryFrom<Payload> for image_generation::ImageGenerationResponse<Payload> {
        type Error = ImageGenerationError;

        fn try_from(_: Payload) -> Result<Self, Self::Error> {
            unreachable!("a 429 never reaches payload conversion")
        }
    }

    #[tokio::test]
    async fn non_success_response_preserves_headers() {
        let mut headers = http::HeaderMap::new();
        headers.insert(http::header::RETRY_AFTER, "20".parse().expect("value"));
        let client = RecordingHttpClient::with_error_response_headers(
            http::StatusCode::TOO_MANY_REQUESTS,
            r#"{"error":"slow down"}"#,
            headers,
        );

        let error = send_image_generation::<_, DirectPayload<Payload>>(
            &client,
            http_client::Request::builder()
                .method(http::Method::POST)
                .uri("https://example.test/v1/images/generations"),
            serde_json::json!({}),
        )
        .await
        .err()
        .expect("a 429 should fail");

        assert_eq!(
            error
                .provider_response_headers()
                .and_then(|headers| headers.get(http::header::RETRY_AFTER))
                .and_then(|value| value.to_str().ok()),
            Some("20"),
        );
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::TOO_MANY_REQUESTS)
        );
    }
}
