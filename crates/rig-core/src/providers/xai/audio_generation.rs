use crate::audio_generation::{
    AudioGenerationError, AudioGenerationRequest, AudioGenerationResponse,
};
use crate::json_utils::merge_inplace;
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

    let mut body = json!({
        "text": request.text,
        "voice_id": voice,
        "language": "en",
    });

    if let Some(additional_params) = request.additional_params.clone() {
        merge_inplace(&mut body, additional_params);
    }

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
