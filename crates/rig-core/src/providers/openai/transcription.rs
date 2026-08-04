use crate::transcription;
use crate::transcription::TranscriptionError;
use serde::Deserialize;

// ================================================================
// OpenAI Transcription API
// ================================================================

pub const WHISPER_1: &str = "whisper-1";

#[derive(Debug, Deserialize)]
pub struct TranscriptionResponse {
    pub text: String,
}

impl TryFrom<TranscriptionResponse>
    for transcription::TranscriptionResponse<TranscriptionResponse>
{
    type Error = TranscriptionError;

    fn try_from(value: TranscriptionResponse) -> Result<Self, Self::Error> {
        Ok(transcription::TranscriptionResponse {
            text: value.text.clone(),
            response: value,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http_runtime::HttpRuntime;
    use crate::providers::openai::functions;
    use crate::test_utils::RecordingHttpClient;
    use crate::transcription::TranscriptionRequest;

    #[tokio::test]
    async fn transcription_http_non_success_preserves_status_and_body() {
        let body = r#"{"error":{"message":"bad audio","type":"invalid_request_error"}}"#;
        let rt = HttpRuntime::recording(RecordingHttpClient::with_error_response(
            http::StatusCode::BAD_REQUEST,
            body,
        ));
        let cfg = functions::Config::new(WHISPER_1).with_api_key("test-key");

        let Err(error) =
            functions::transcribe(&cfg, &rt, TranscriptionRequest::new(vec![0u8; 16])).await
        else {
            panic!("transcription should fail with non-success status");
        };

        assert!(matches!(error, TranscriptionError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::BAD_REQUEST)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }

    #[tokio::test]
    async fn transcription_success_parses_text() {
        let rt = HttpRuntime::recording(RecordingHttpClient::new(r#"{"text":"hello world"}"#));
        let cfg = functions::Config::new(WHISPER_1).with_api_key("test-key");

        let response = functions::transcribe(&cfg, &rt, TranscriptionRequest::new(vec![0u8; 4]))
            .await
            .expect("transcription should succeed");

        assert_eq!(response.text, "hello world");
    }
}
