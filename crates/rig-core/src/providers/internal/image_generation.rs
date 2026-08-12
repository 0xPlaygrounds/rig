//! Shared request driver for OpenAI-style image generation endpoints.
//!
//! OpenAI, Azure OpenAI, xAI and Hyperbolic each build a provider-specific
//! JSON body, but the tail is identical: POST the body, classify the response
//! through the provider's success-or-error envelope, and convert the payload.
//! The body and path stay with each provider — those are the real wire
//! differences — while this driver owns the shared send/decode tail.

use bytes::Bytes;
use serde::de::DeserializeOwned;

use super::envelope::ProviderEnvelope;
use crate::http_client::{self, HttpClientExt};
use crate::image_generation::{self, ImageGenerationError};

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

    let status = response.status();
    let response_body = response.into_body().into_future().await?;

    if !status.is_success() {
        return Err(ImageGenerationError::from_http_response(
            status,
            String::from_utf8_lossy(&response_body).into_owned(),
        ));
    }

    match serde_json::from_slice::<A>(&response_body)?.into_payload() {
        Ok(response) => response.try_into(),
        Err(message) => {
            tracing::warn!(message = %message, "provider returned an error response");
            Err(ImageGenerationError::from_http_response(
                status,
                String::from_utf8_lossy(&response_body).into_owned(),
            ))
        }
    }
}
