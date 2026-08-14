//! Shared request driver for JSON audio-generation endpoints.
//!
//! Providers build their own JSON body and path, then share the identical raw
//! audio response and error-preservation tail.

use bytes::Bytes;

use crate::audio_generation::{
    self, AudioGenerationError, AudioGenerationRequest, AudioGenerationResponse,
};
use crate::client::{Client, Provider};
use crate::http_client::{self, HttpClientExt};
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};

/// Provider-specific request construction for the shared raw-audio model.
pub(crate) trait RawAudioGenerationProvider: Provider {
    const AUDIO_GENERATION_PATH: &'static str;
    const EXPLICIT_JSON_CONTENT_TYPE: bool = false;

    fn audio_generation_request_builder<H>(
        client: &Client<Self, H>,
        _model: &str,
    ) -> Result<http_client::Builder, AudioGenerationError>
    where
        H: HttpClientExt,
    {
        let builder = client.post(Self::AUDIO_GENERATION_PATH)?;
        Ok(if Self::EXPLICIT_JSON_CONTENT_TYPE {
            builder.header("Content-Type", "application/json")
        } else {
            builder
        })
    }

    fn audio_generation_request_body(
        model: &str,
        request: AudioGenerationRequest,
    ) -> Result<serde_json::Value, AudioGenerationError> {
        Ok(serde_json::json!({
            "model": model,
            "input": request.text,
            "voice": request.voice,
            "speed": request.speed,
        }))
    }
}

/// Shared model shell for providers whose audio endpoint returns raw bytes.
///
/// Public provider modules expose this through their own `AudioGenerationModel`
/// aliases; request routing and JSON shape remain on the provider extension.
#[doc(hidden)]
#[derive(Clone)]
pub struct GenericAudioGenerationModel<Ext, H = reqwest::Client> {
    client: Client<Ext, H>,
    /// Name of the audio generation model.
    pub model: String,
}

impl<Ext, H> GenericAudioGenerationModel<Ext, H> {
    /// Creates an audio generation model backed by `client`.
    pub fn new(client: Client<Ext, H>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }
}

impl<Ext, H> audio_generation::AudioGenerationModel for GenericAudioGenerationModel<Ext, H>
where
    Ext: RawAudioGenerationProvider + Clone + WasmCompatSend + WasmCompatSync + 'static,
    H: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    type Response = Bytes;
    type Client = Client<Ext, H>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model)
    }

    async fn audio_generation(
        &self,
        request: AudioGenerationRequest,
    ) -> Result<AudioGenerationResponse<Self::Response>, AudioGenerationError> {
        let builder = Ext::audio_generation_request_builder(&self.client, &self.model)?;
        let body = Ext::audio_generation_request_body(&self.model, request)?;
        send_audio_generation(&self.client, builder, body).await
    }
}

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
