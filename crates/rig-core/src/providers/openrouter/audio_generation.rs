use crate::audio_generation::{AudioGenerationError, AudioGenerationRequest};
use crate::providers::internal::audio_generation::{
    GenericAudioGenerationModel, RawAudioGenerationProvider,
};
use crate::providers::openrouter::OpenRouterExt;
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

/// OpenRouter audio generation model.
pub type AudioGenerationModel<T = reqwest::Client> = GenericAudioGenerationModel<OpenRouterExt, T>;

impl RawAudioGenerationProvider for OpenRouterExt {
    const AUDIO_GENERATION_PATH: &'static str = "/audio/speech";
    const EXPLICIT_JSON_CONTENT_TYPE: bool = true;

    fn audio_generation_request_body(
        model: &str,
        request: AudioGenerationRequest,
    ) -> Result<serde_json::Value, AudioGenerationError> {
        let mut body_map: serde_json::Map<String, serde_json::Value> = [
            ("model".to_string(), json!(model)),
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

        Ok(serde_json::Value::Object(body_map))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_generation::AudioGenerationModel as _;
    use crate::client::audio_generation::AudioGenerationClient;
    use crate::providers::openrouter::Client;
    use crate::test_utils::RecordingHttpClient;
    use bytes::Bytes;

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
