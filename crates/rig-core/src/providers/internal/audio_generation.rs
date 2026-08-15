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

    // Taking the response apart hands the headers over already owned, so a
    // failure keeps its rate-limit metadata at no cost to the success path
    // (rig#2210).
    let (parts, body) = response.into_parts();
    let status = parts.status;
    let bytes: Bytes = body.await?;

    if !status.is_success() {
        return Err(AudioGenerationError::from_http_response(
            status,
            String::from_utf8_lossy(&bytes),
        )
        .with_response_headers(Some(Box::new(parts.headers))));
    }

    Ok(AudioGenerationResponse {
        audio: bytes.to_vec(),
        response: bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The default body every provider inherits unless it overrides it.
    struct DefaultAudioExt;

    impl Provider for DefaultAudioExt {
        type Builder = crate::providers::openai::OpenAICompletionsExtBuilder;
        const VERIFY_PATH: &'static str = "/models";
    }

    impl RawAudioGenerationProvider for DefaultAudioExt {
        const AUDIO_GENERATION_PATH: &'static str = "/audio/speech";
    }

    fn body(additional_params: Option<serde_json::Value>) -> serde_json::Value {
        DefaultAudioExt::audio_generation_request_body(
            "tts-1",
            AudioGenerationRequest {
                text: "hello".to_owned(),
                voice: "alloy".to_owned(),
                speed: 1.0,
                additional_params,
            },
        )
        .expect("body should build")
    }

    #[test]
    fn request_body_derives_the_documented_fields() {
        let body = body(None);

        assert_eq!(body["model"], json!("tts-1"));
        assert_eq!(body["input"], json!("hello"));
        assert_eq!(body["voice"], json!("alloy"));
        assert_eq!(body["speed"], json!(1.0));
    }

    /// The defect this default carried: the field reached no provider that
    /// inherited the body, even though the endpoint acts on it.
    #[test]
    fn request_body_merges_additional_params_last() {
        let body = body(Some(
            json!({ "response_format": "wav", "instructions": "Speak slowly." }),
        ));

        assert_eq!(body["response_format"], json!("wav"));
        assert_eq!(body["instructions"], json!("Speak slowly."));
    }

    /// Merged last, so a caller can override a derived key.
    #[test]
    fn request_body_lets_additional_params_override_derived_keys() {
        let body = body(Some(
            json!({ "voice": "nova", "speed": 0.5, "input": "other" }),
        ));

        assert_eq!(body["voice"], json!("nova"));
        assert_eq!(body["speed"], json!(0.5));
        assert_eq!(body["input"], json!("other"));
    }

    #[test]
    fn request_body_ignores_non_object_additional_params() {
        assert_eq!(body(Some(json!("not-an-object"))), body(None));
        assert_eq!(body(Some(json!(null))), body(None));
    }
}

/// rig#2210: a failed audio-generation response keeps its headers, so the
/// capability error's `provider_response_headers()` is not a promise the
/// driver quietly breaks.
#[cfg(test)]
mod header_preservation_tests {
    use super::*;
    use crate::test_utils::RecordingHttpClient;

    #[tokio::test]
    async fn non_success_response_preserves_headers() {
        let mut headers = http::HeaderMap::new();
        headers.insert(http::header::RETRY_AFTER, "20".parse().expect("value"));
        let client = RecordingHttpClient::with_error_response_headers(
            http::StatusCode::TOO_MANY_REQUESTS,
            r#"{"error":"slow down"}"#,
            headers,
        );

        let error = send_audio_generation(
            &client,
            http_client::Request::builder()
                .method(http::Method::POST)
                .uri("https://example.test/v1/audio/speech"),
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
