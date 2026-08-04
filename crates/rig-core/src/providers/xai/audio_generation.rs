use crate::audio_generation::{
    AudioGenerationError, AudioGenerationRequest, AudioGenerationResponse,
};
use bytes::Bytes;
use serde_json::json;

// ================================================================
// xAI TTS API
// ================================================================
pub const TTS_1: &str = "tts-1";

/// Build the serialized TTS request body. Pure.
///
/// xAI's TTS route has no model field; an empty voice falls back to `eve`.
pub(crate) fn build_audio_generation_body(
    request: &AudioGenerationRequest,
) -> Result<Vec<u8>, AudioGenerationError> {
    let voice = if request.voice.is_empty() {
        "eve"
    } else {
        request.voice.as_str()
    };

    let body = json!({
        "text": request.text,
        "voice_id": voice,
        "speed": request.speed,
    });
    let mut body = crate::json_utils::merge_additional_params(
        body,
        request.additional_params.clone(),
        &["text", "voice_id", "speed"],
        "xAI audio-generation request",
    )?;
    body.as_object_mut()
        .ok_or(crate::json_utils::RequestOverlayError::CanonicalNotObject {
            context: "xAI audio-generation request",
        })?
        .entry("language".to_string())
        .or_insert_with(|| json!("en"));

    Ok(serde_json::to_vec(&body)?)
}

/// Parse a TTS response: success bodies are raw audio bytes. Pure.
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
    use crate::providers::xai::functions;

    #[test]
    fn audio_generation_allows_provider_native_language() {
        let mut request = AudioGenerationRequest::new("hola", "eve");
        request.additional_params = Some(serde_json::json!({"language": "es-MX"}));

        let body = build_audio_generation_body(&request)
            .expect("provider-native language should override the fallback");
        let value: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON");

        assert_eq!(value["language"], "es-MX");
    }

    #[test]
    fn audio_generation_defaults_language_when_omitted() {
        let request = AudioGenerationRequest::new("hello", "eve");
        let body = build_audio_generation_body(&request).expect("request should serialize");
        let value: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON");

        assert_eq!(value["language"], "en");
    }

    #[test]
    fn audio_generation_serializes_generic_speed() {
        let request = AudioGenerationRequest::new("hello", "eve").with_speed(1.2);
        let body = build_audio_generation_body(&request).expect("request should serialize");
        let value: serde_json::Value = serde_json::from_slice(&body).expect("valid JSON");

        let speed = value["speed"].as_f64().expect("speed should be numeric");
        assert_eq!(speed, f64::from(1.2_f32));

        let colliding = request.with_additional_params(serde_json::json!({"speed": 0.8}));
        let error = build_audio_generation_body(&colliding)
            .expect_err("request-owned speed must not be replaced");
        assert!(matches!(error, AudioGenerationError::RequestError(_)));
        assert!(error.to_string().contains("speed"));
    }

    #[tokio::test]
    async fn audio_generation_non_success_preserves_status_and_body() {
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"error":"boom","code":"503"}"#;
        let rt = crate::http_runtime::HttpRuntime::recording(
            RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body),
        );
        let cfg = functions::Config::new(TTS_1).with_api_key("test-key");

        let request = AudioGenerationRequest::new("hello", "eve");

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
