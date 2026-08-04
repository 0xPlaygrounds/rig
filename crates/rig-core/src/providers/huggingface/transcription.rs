use crate::providers::huggingface::completion::ApiResponse;
use crate::transcription;
use crate::transcription::TranscriptionError;
use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use serde::Deserialize;
use serde_json::json;

pub const WHISPER_LARGE_V3: &str = "openai/whisper-large-v3";
pub const WHISPER_LARGE_V3_TURBO: &str = "openai/whisper-large-v3-turbo";

pub const WHISPER_SMALL: &str = "openai/whisper-small";

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

/// Build the serialized transcription request body (base64 audio). Pure.
pub(crate) fn build_transcription_body(data: &[u8]) -> Result<Vec<u8>, TranscriptionError> {
    Ok(serde_json::to_vec(&json!({
        "inputs": BASE64_STANDARD.encode(data)
    }))?)
}

/// Parse a transcription response body. Pure.
pub(crate) fn parse_transcription_response(
    status: http::StatusCode,
    body: &[u8],
) -> Result<transcription::TranscriptionResponse<TranscriptionResponse>, TranscriptionError> {
    if !status.is_success() {
        return Err(TranscriptionError::from_http_response(
            status,
            String::from_utf8_lossy(body),
        ));
    }

    match serde_json::from_slice::<ApiResponse<TranscriptionResponse>>(body)? {
        ApiResponse::Ok(response) => response.try_into(),
        ApiResponse::Err(err) => {
            let message = err
                .get("error")
                .and_then(|e| {
                    e.as_str()
                        .or_else(|| e.get("message").and_then(|m| m.as_str()))
                })
                .or_else(|| err.get("message").and_then(|m| m.as_str()))
                .unwrap_or_default();
            tracing::warn!(message = %message, "provider returned an error response");
            Err(TranscriptionError::from_http_response(
                status,
                String::from_utf8_lossy(body),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http_runtime::HttpRuntime;
    use crate::providers::huggingface::functions;
    use crate::test_utils::RecordingHttpClient;
    use crate::transcription::TranscriptionRequest;

    #[tokio::test]
    async fn transcription_non_success_preserves_status_and_body() {
        let body = r#"{"error":{"message":"boom"}}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
        let rt = HttpRuntime::recording(http_client);
        let cfg = functions::Config::new(WHISPER_LARGE_V3).with_api_key("test-key");

        let error = functions::transcribe(&cfg, &rt, TranscriptionRequest::new(vec![0u8; 16]))
            .await
            .err()
            .expect("should fail with non-success status");

        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::SERVICE_UNAVAILABLE)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }

    #[tokio::test]
    async fn transcription_2xx_error_envelope_preserves_status_and_body() {
        // A 200 OK body that is not a valid `TranscriptionResponse` (no `text`
        // field) falls through the untagged `ApiResponse` to its `Err(Value)`
        // variant, which the provider routes through `from_http_response`.
        let body = r#"{"error":"Model openai/whisper-large-v3 is currently loading"}"#;
        let rt = HttpRuntime::recording(RecordingHttpClient::new(body));
        let cfg = functions::Config::new(WHISPER_LARGE_V3).with_api_key("test-key");

        let error = functions::transcribe(&cfg, &rt, TranscriptionRequest::new(vec![0u8; 16]))
            .await
            .err()
            .expect("should fail with provider error envelope");

        match &error {
            TranscriptionError::ProviderResponse(stored) => {
                assert_eq!(stored.body, body);
                assert_eq!(stored.status, Some(http::StatusCode::OK));
            }
            other => panic!("expected ProviderResponse, got {other:?}"),
        }
    }
}
