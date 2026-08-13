//! Shared request driver for JSON audio-generation endpoints.
//!
//! Providers build their own JSON body and path, then share the identical raw
//! audio response and error-preservation tail.

use bytes::Bytes;

use crate::audio_generation::{AudioGenerationError, AudioGenerationResponse};
use crate::http_client::{self, HttpClientExt};

/// Sends an audio generation request and returns the raw audio bytes.
///
/// `builder` is the provider's already-path-built POST request; `body` is the
/// provider's JSON request body. Provider error bodies are preserved raw via
/// [`AudioGenerationError::from_http_response`].
pub(crate) async fn send_audio_generation<C>(
    client: &C,
    builder: http_client::Builder,
    body: serde_json::Value,
) -> Result<AudioGenerationResponse<Bytes>, AudioGenerationError>
where
    C: HttpClientExt,
{
    let body = serde_json::to_vec(&body)?;

    let req = builder
        .body(body)
        .map_err(|e| AudioGenerationError::HttpError(e.into()))?;

    let response = client.send::<_, Bytes>(req).await?;

    let status = response.status();
    let bytes: Bytes = response.into_body().await?;

    if !status.is_success() {
        return Err(AudioGenerationError::from_http_response(
            status,
            String::from_utf8_lossy(&bytes),
        ));
    }

    Ok(AudioGenerationResponse {
        audio: bytes.to_vec(),
        response: bytes,
    })
}
