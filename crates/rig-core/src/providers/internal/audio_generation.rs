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
#[doc(hidden)]
pub trait RawAudioGenerationProvider: Provider {
    const AUDIO_GENERATION_PATH: &'static str;
    const EXPLICIT_JSON_CONTENT_TYPE: bool = false;

    /// Stable descriptor name of the provider, stamped on every normalized
    /// response — an input to normalization, never hardcoded in the driver.
    const PROVIDER_NAME: &'static str;

    /// The provider's transport request-id response header, when it has one.
    const REQUEST_ID_HEADER: Option<&'static str> = None;

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
        let mut body = serde_json::json!({
            "model": model,
            "input": request.text,
            "voice": request.voice,
            "speed": request.speed,
        });

        // Last, so a caller can reach the endpoint's other parameters —
        // `response_format`, `instructions` — and override what is derived
        // above. Every provider that overrides this body already merges the
        // field (xAI, OpenRouter, Venice); leaving it out of the *default*
        // made `AudioGenerationRequestBuilder::additional_params` silently
        // inert for whoever inherited it, OpenAI included, even though the
        // parameters demonstrably change the response (`response_format:
        // "wav"` returns RIFF where the default returns MP3).
        if let Some(additional_params) = request.additional_params {
            crate::json_utils::merge_inplace(&mut body, additional_params);
        }

        Ok(body)
    }
}

/// Shared model shell for providers whose audio endpoint returns raw bytes.
///
/// Public provider modules expose this through their own `AudioGenerationModel`
/// aliases; request routing and JSON shape remain on the provider extension.
#[doc(hidden)]
#[derive(Clone)]
pub struct GenericAudioGenerationModel<Ext, H = crate::http_client::BoxedHttpClient> {
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

impl<Ext, H> GenericAudioGenerationModel<Ext, H>
where
    Ext: RawAudioGenerationProvider + Clone + WasmCompatSend + WasmCompatSync + 'static,
    H: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Perform the generation and return the provider's native response —
    /// the audio bytes as sent, these endpoints answer with no JSON envelope
    /// — instead of the normalized [`AudioGenerationResponse`]. Same request,
    /// transport, and error path as
    /// [`audio_generation::AudioGenerationModel::audio_generation`].
    pub async fn raw_audio_generation(
        &self,
        request: AudioGenerationRequest,
    ) -> Result<Bytes, AudioGenerationError> {
        self.raw_audio_generation_with_request_id(request)
            .await
            .map(|(bytes, _)| bytes)
    }

    /// [`Self::raw_audio_generation`] plus the transport request id from the
    /// provider's request-id response header, when it carries one.
    pub async fn raw_audio_generation_with_request_id(
        &self,
        request: AudioGenerationRequest,
    ) -> Result<(Bytes, Option<String>), AudioGenerationError> {
        let builder = Ext::audio_generation_request_builder(&self.client, &self.model)?;
        let body = Ext::audio_generation_request_body(&self.model, request)?;
        send_audio_generation(&self.client, builder, body, Ext::REQUEST_ID_HEADER).await
    }
}

impl<Ext, H> audio_generation::AudioGenerationModel for GenericAudioGenerationModel<Ext, H>
where
    Ext: RawAudioGenerationProvider + Clone + WasmCompatSend + WasmCompatSync + 'static,
    H: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    async fn audio_generation(
        &self,
        request: AudioGenerationRequest,
    ) -> Result<AudioGenerationResponse, AudioGenerationError> {
        crate::telemetry::instrument_modality(
            Ext::PROVIDER_NAME,
            &self.model,
            crate::telemetry::ModalityOperation::AudioGeneration,
            async {
                let (bytes, provider_request_id) =
                    self.raw_audio_generation_with_request_id(request).await?;
                // The native response is bytes, not JSON: `raw` stays `Null` and the
                // typed route is `raw_audio_generation`.
                Ok(
                    AudioGenerationResponse::new(bytes.to_vec(), Ext::PROVIDER_NAME)
                        .with_optional_provider_request_id(provider_request_id),
                )
            },
        )
        .await
    }
}

impl<Ext, H> crate::client::ConstructAudioGenerationModel<Client<Ext, H>>
    for GenericAudioGenerationModel<Ext, H>
where
    Ext: RawAudioGenerationProvider + Clone + WasmCompatSend + WasmCompatSync + 'static,
    H: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    fn construct(client: &Client<Ext, H>, model: String) -> Self {
        Self::new(client.clone(), model)
    }
}

/// Sends an audio generation request and returns the raw audio bytes plus the
/// transport request id read from `request_id_header`, when the provider has
/// one and the response carried it.
///
/// `builder` is the provider's already-path-built POST request; `body` is the
/// provider's JSON request body. Provider error bodies are preserved raw via
/// [`AudioGenerationError::from_http_response`].
pub(crate) async fn send_audio_generation<C>(
    client: &C,
    builder: http_client::Builder,
    body: serde_json::Value,
    request_id_header: Option<&str>,
) -> Result<(Bytes, Option<String>), AudioGenerationError>
where
    C: HttpClientExt,
{
    let body = serde_json::to_vec(&body)?;

    let req = builder
        .body(body)
        .map_err(|e| AudioGenerationError::HttpError(e.into()))?;

    let response = client.send::<_, Bytes>(req).await?;

    // Taking the response apart hands the headers over already owned, so a
    // failure keeps its rate-limit metadata at no cost to the success path
    // (rig#2210).
    let (parts, body) = response.into_parts();
    let status = parts.status;
    let provider_request_id =
        super::transcription::request_id_from_headers(&parts.headers, request_id_header);
    let bytes: Bytes = body.await?;

    if !status.is_success() {
        return Err(AudioGenerationError::from_http_response(
            status,
            String::from_utf8_lossy(&bytes),
        )
        .with_response_headers(Some(Box::new(parts.headers))));
    }

    Ok((bytes, provider_request_id))
}

#[cfg(test)]
mod tests;

/// rig#2210: a failed audio-generation response keeps its headers, so the
/// capability error's `provider_response_headers()` is not a promise the
/// driver quietly breaks.
#[cfg(test)]
mod header_preservation_tests;
