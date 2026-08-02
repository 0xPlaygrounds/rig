use crate::audio_generation::{
    AudioGenerationError, AudioGenerationRequest, AudioGenerationResponse,
};
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

/// Build the serialized audio-generation (TTS) request body. Pure.
pub(crate) fn build_audio_generation_body(
    model: &str,
    request: &AudioGenerationRequest,
) -> Result<Vec<u8>, AudioGenerationError> {
    let mut body_map: serde_json::Map<String, serde_json::Value> = [
        ("model".to_string(), json!(model)),
        ("input".to_string(), json!(request.text)),
        ("voice".to_string(), json!(request.voice)),
        ("speed".to_string(), json!(request.speed)),
    ]
    .into_iter()
    .collect();

    if let Some(params) = crate::json_utils::validated_additional_params(
        request.additional_params.as_ref(),
        &["model", "input", "voice", "speed"],
        "OpenRouter audio-generation request",
    )? {
        for (k, v) in params {
            body_map.insert(k.clone(), v.clone());
        }
    }
    body_map
        .entry("response_format".to_string())
        .or_insert_with(|| json!("mp3"));

    Ok(serde_json::to_vec(&serde_json::Value::Object(body_map))?)
}

/// Parse an audio-generation response: success bodies are raw audio bytes.
/// Pure.
pub(crate) fn parse_audio_generation_response(
    status: http::StatusCode,
    body: Vec<u8>,
) -> Result<AudioGenerationResponse<Bytes>, AudioGenerationError> {
    if !status.is_success() {
        return Err(AudioGenerationError::from_http_response(
            status,
            String::from_utf8_lossy(&body).into_owned(),
        ));
    }
    Ok(AudioGenerationResponse {
        audio: body.clone(),
        response: Bytes::from(body),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::openrouter::functions;
    use crate::test_utils::RecordingHttpClient;

    #[test]
    fn audio_generation_allows_provider_native_response_format() {
        let mut request = AudioGenerationRequest::new("hello", "alloy");
        request.additional_params = Some(serde_json::json!({"response_format": "pcm"}));

        let body = build_audio_generation_body(GPT_4O_MINI_TTS, &request)
            .expect("provider-native response format should override the fallback");
        let value: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON");

        assert_eq!(value["response_format"], "pcm");
    }

    #[test]
    fn audio_generation_still_rejects_request_owned_collisions() {
        let mut request = AudioGenerationRequest::new("hello", "alloy");
        request.additional_params = Some(serde_json::json!({"voice": "echo"}));

        let error = build_audio_generation_body(GPT_4O_MINI_TTS, &request)
            .expect_err("request-owned voice must not be replaced");

        assert!(matches!(error, AudioGenerationError::RequestError(_)));
        assert!(error.to_string().contains("voice"));
    }

    #[tokio::test]
    async fn audio_generation_non_success_preserves_status_and_body() {
        let body = r#"{"error":{"message":"boom"}}"#;
        let rt = crate::http_runtime::HttpRuntime::recording(
            RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body),
        );
        let cfg = functions::Config::new(GPT_4O_MINI_TTS).with_api_key("test-key");

        let request = AudioGenerationRequest::new("hello", "alloy");

        let error = functions::generate_audio(&cfg, &rt, request)
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
