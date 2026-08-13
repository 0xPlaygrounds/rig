use crate::audio_generation::{
    self, AudioGenerationError, AudioGenerationRequest, AudioGenerationResponse,
};
use crate::http_client::HttpClientExt;
use crate::providers::internal::audio_generation::send_audio_generation;
use crate::providers::openrouter::Client;
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use bytes::Bytes;
use serde_json::json;

// ================================================================
// Model constants
// ================================================================

/// The `openai/gpt-4o-mini-tts-2025-12-15` model.
pub const GPT_4O_MINI_TTS: &str = "openai/gpt-4o-mini-tts-2025-12-15";
/// The `mistralai/voxtral-mini-tts-2603` model.
pub const VOXTRAL_MINI_TTS: &str = "mistralai/voxtral-mini-tts-2603";
/// The `hexgrad/kokoro-82m` model.
pub const KOKORO_82M: &str = "hexgrad/kokoro-82m";

// ================================================================
// Model
// ================================================================

#[derive(Clone)]
pub struct AudioGenerationModel<T = reqwest::Client> {
    client: Client<T>,
    pub model: String,
}

impl<T> AudioGenerationModel<T> {
    pub fn new(client: Client<T>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }
}

impl<T> audio_generation::AudioGenerationModel for AudioGenerationModel<T>
where
    T: HttpClientExt
        + Clone
        + std::fmt::Debug
        + Default
        + WasmCompatSend
        + WasmCompatSync
        + 'static,
{
    type Response = Bytes;
    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model)
    }

    async fn audio_generation(
        &self,
        request: AudioGenerationRequest,
    ) -> Result<AudioGenerationResponse<Self::Response>, AudioGenerationError> {
        let mut body_map: serde_json::Map<String, serde_json::Value> = [
            ("model".to_string(), json!(self.model)),
            ("input".to_string(), json!(request.text)),
            ("voice".to_string(), json!(request.voice)),
            ("response_format".to_string(), json!("mp3")),
            ("speed".to_string(), json!(request.speed)),
        ]
        .into_iter()
        .collect();

        if let Some(ref additional_params) = request.additional_params {
            let params = additional_params.as_object().ok_or_else(|| {
                AudioGenerationError::RequestError(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "additional audio generation parameters must be a JSON object",
                )))
            })?;
            for (k, v) in params {
                body_map.insert(k.clone(), v.clone());
            }
        }

        let builder = self
            .client
            .post("/audio/speech")?
            .header("Content-Type", "application/json");
        send_audio_generation(&self.client, builder, serde_json::Value::Object(body_map)).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_generation::AudioGenerationModel as _;
    use crate::client::audio_generation::AudioGenerationClient;
    use crate::test_utils::RecordingHttpClient;

    #[tokio::test]
    async fn shared_driver_keeps_openrouter_request_and_binary_response() {
        let http_client = RecordingHttpClient::new(Bytes::from_static(b"audio"));
        let client = Client::builder()
            .api_key("test-key")
            .http_client(http_client.clone())
            .build()
            .expect("build client");
        let model = client.audio_generation_model(GPT_4O_MINI_TTS);

        let response = model
            .audio_generation(
                model
                    .audio_generation_request()
                    .text("hello")
                    .voice("alloy")
                    .build(),
            )
            .await
            .expect("audio generation should succeed");

        assert_eq!(response.audio, b"audio");
        let requests = http_client.requests();
        assert_eq!(requests[0].uri, "https://openrouter.ai/api/v1/audio/speech");
        assert_eq!(
            requests[0]
                .headers
                .get(http::header::CONTENT_TYPE)
                .and_then(|value| value.to_str().ok()),
            Some("application/json")
        );
        let body: serde_json::Value =
            serde_json::from_slice(&requests[0].body).expect("request body should be JSON");
        assert_eq!(body["model"], GPT_4O_MINI_TTS);
        assert_eq!(body["input"], "hello");
        assert_eq!(body["voice"], "alloy");
    }

    #[tokio::test]
    async fn audio_generation_non_success_preserves_status_and_body() {
        let body = r#"{"error":{"message":"boom"}}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
        let client = Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.audio_generation_model(GPT_4O_MINI_TTS);

        let request = model
            .audio_generation_request()
            .text("hello")
            .voice("alloy")
            .build();

        let error = model
            .audio_generation(request)
            .await
            .err()
            .expect("should fail with non-success status");

        assert!(matches!(error, AudioGenerationError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::SERVICE_UNAVAILABLE)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }
}
