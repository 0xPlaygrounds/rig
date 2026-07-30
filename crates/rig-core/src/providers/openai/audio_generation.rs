//! OpenAI audio generation (text-to-speech) model identifiers.
//!
//! The request/response wire handling lives in
//! [`super::functions`] (`build_audio_generation_body`,
//! `parse_audio_generation_response`, `generate_audio`).

pub const TTS_1: &str = "tts-1";
pub const TTS_1_HD: &str = "tts-1-hd";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_generation::{AudioGenerationError, AudioGenerationRequest};
    use crate::http_runtime::HttpRuntime;
    use crate::providers::openai::functions;
    use crate::test_utils::RecordingHttpClient;

    #[tokio::test]
    async fn audio_generation_non_success_preserves_status_and_body() {
        let body = r#"{"error":{"message":"boom"}}"#;
        let rt = HttpRuntime::recording(RecordingHttpClient::with_error_response(
            http::StatusCode::SERVICE_UNAVAILABLE,
            body,
        ));
        let cfg = functions::Config::new(TTS_1).with_api_key("test-key");

        let Err(error) =
            functions::generate_audio(&cfg, &rt, AudioGenerationRequest::new("hello", "alloy"))
                .await
        else {
            panic!("should fail with non-success status");
        };

        assert!(matches!(error, AudioGenerationError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::SERVICE_UNAVAILABLE)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }

    #[tokio::test]
    async fn audio_generation_success_returns_raw_audio_bytes() {
        let rt = HttpRuntime::recording(RecordingHttpClient::new("RIFFfake-audio"));
        let cfg = functions::Config::new(TTS_1_HD).with_api_key("test-key");

        let response =
            functions::generate_audio(&cfg, &rt, AudioGenerationRequest::new("hello", "alloy"))
                .await
                .expect("audio generation should succeed");

        assert_eq!(response.audio, b"RIFFfake-audio".to_vec());
    }
}
