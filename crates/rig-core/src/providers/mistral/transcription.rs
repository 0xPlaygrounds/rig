//! Implements Mistral (basic) transcription API
use bytes::Bytes;
use serde::{Deserialize, Serialize};

use crate::completion::Usage;
use crate::http_client::HttpClientExt;
use crate::providers::internal::transcription::request_id_from_headers;
use crate::providers::internal::transcription::{TranscriptionFields, transcription_form};
use crate::providers::mistral::Client;
use crate::transcription::{self, NormalizeTranscriptionResponse, TranscriptionError};
use crate::wasm_compat::WasmCompatSend;

// ================================================================
// Mistral Transcription API
// ================================================================

/// Voxtral Mini model (latest version)
pub const VOXTRAL_MINI: &str = "voxtral-mini-latest";
/// Voxtral Small model (latest version)
pub const VOXTRAL_SMALL: &str = "voxtral-small-latest";

/// Request usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionUsage {
    pub prompt_audio_seconds: Option<i32>,
    pub prompt_tokens: i32,
    pub total_tokens: i32,
    pub completion_tokens: i32,
    pub prompt_tokens_details: Option<serde_json::Value>,
}

impl std::fmt::Display for TranscriptionUsage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Usage:")?;
        writeln!(f, "  prompt_tokens:     {}", self.prompt_tokens)?;
        writeln!(f, "  completion_tokens: {}", self.completion_tokens)?;
        writeln!(f, "  total_tokens:      {}", self.total_tokens)?;
        if let Some(details) = &self.prompt_tokens_details {
            writeln!(f, "  prompt_token_details: {:?}", details)?;
        } else {
            writeln!(f, "  prompt_token_details: N/A")?;
        }
        if let Some(secs) = self.prompt_audio_seconds {
            write!(f, "  audio_seconds:     {secs}")?;
        } else {
            write!(f, "  audio_seconds:     N/A")?;
        }
        Ok(())
    }
}

/// Diarization information, tells when each speaker started and ended talking plus what they said.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentChunk {
    /// Start time in seconds
    pub start: f32,
    /// End time in seconds
    pub end: f32,
    /// Segment transcribed text
    pub text: String,
    pub score: Option<f32>,
    /// Speaker identification.
    pub speaker_id: Option<String>,
    #[serde(rename = "type")]
    pub segment_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralTranscriptionResponse {
    /// Audio language
    pub language: Option<String>,
    /// Model name (e.g. voxtra-mini-latest)
    pub model: String,
    /// An array of transcript segments, each containing a portion of the transcribed text along with its start and end times in seconds and speaker id (if diarization was enabled).
    pub segments: Vec<SegmentChunk>,
    /// Audio Transcription
    pub text: String,
    /// Request token usage statistics
    pub usage: TranscriptionUsage,
}

impl NormalizeTranscriptionResponse for MistralTranscriptionResponse {
    fn normalize(
        self,
        provider: &str,
    ) -> Result<transcription::TranscriptionResponse, TranscriptionError> {
        let usage = Usage {
            input_tokens: self.usage.prompt_tokens.max(0) as u64,
            output_tokens: self.usage.completion_tokens.max(0) as u64,
            total_tokens: self.usage.total_tokens.max(0) as u64,
            ..Usage::new()
        };
        Ok(
            transcription::TranscriptionResponse::new(self.text, provider)
                .with_model(self.model)
                .with_usage(usage),
        )
    }
}

pub type TranscriptionModel<T = reqwest::Client> =
    crate::providers::internal::transcription::GenericTranscriptionModel<
        crate::providers::mistral::client::MistralExt,
        T,
    >;

impl<T> TranscriptionModel<T>
where
    T: HttpClientExt + Clone + WasmCompatSend + 'static,
{
    /// Perform the transcription and return Mistral's native response instead
    /// of the normalized [`transcription::TranscriptionResponse`]. Same
    /// request, transport, parser, and error path as
    /// [`transcription::TranscriptionModel::transcription`].
    pub async fn raw_transcription(
        &self,
        request: transcription::TranscriptionRequest,
    ) -> Result<MistralTranscriptionResponse, TranscriptionError> {
        self.raw_transcription_with_request_id(request)
            .await
            .map(|(response, _)| response)
    }

    /// [`Self::raw_transcription`] plus the `mistral-correlation-id` transport
    /// request id, when the response carried one.
    pub async fn raw_transcription_with_request_id(
        &self,
        mut request: transcription::TranscriptionRequest,
    ) -> Result<(MistralTranscriptionResponse, Option<String>), TranscriptionError> {
        // Mistral's transcription endpoint has no `prompt` field; it has
        // always been dropped rather than sent.
        request.prompt = None;

        let body = transcription_form(
            request,
            TranscriptionFields {
                model: Some(&self.model),
            },
        )?;

        let req = self
            .client
            .post("/v1/audio/transcriptions")?
            .body(body)
            .map_err(|e| TranscriptionError::RequestError(e.into()))?;

        let response = self
            .client
            .send_multipart::<Bytes>(req)
            .await
            .map_err(TranscriptionError::HttpError)?;

        let (parts, body) = response.into_parts();
        let status = parts.status;
        let provider_request_id = request_id_from_headers(
            &parts.headers,
            <super::client::MistralExt as crate::providers::openai::completion::OpenAICompatibleProvider>::REQUEST_ID_HEADER,
        );
        let response_bytes = body.await?;

        if status.is_success() {
            let response_body: MistralTranscriptionResponse =
                serde_json::from_slice(&response_bytes)?;

            tracing::info!(target: "rig", "Mistral transcription token usage: {}", &response_body.usage);

            Ok((response_body, provider_request_id))
        } else {
            Err(TranscriptionError::from_http_response(
                status,
                String::from_utf8_lossy(&response_bytes),
            )
            .with_response_headers(Some(Box::new(parts.headers))))
        }
    }
}

impl<T> transcription::TranscriptionModel for TranscriptionModel<T>
where
    T: HttpClientExt + Clone + WasmCompatSend + 'static,
{
    async fn transcription(
        &self,
        request: transcription::TranscriptionRequest,
    ) -> Result<transcription::TranscriptionResponse, TranscriptionError> {
        crate::telemetry::instrument_modality(
            "mistral",
            &self.model,
            crate::telemetry::ModalityOperation::Transcription,
            async {
                let (response, provider_request_id) =
                    self.raw_transcription_with_request_id(request).await?;
                let captured = serde_json::to_value(&response)?;
                Ok(response
                    .normalize("mistral")?
                    .with_optional_provider_request_id(provider_request_id)
                    .with_raw(captured))
            },
        )
        .await
    }
}

impl<T> crate::client::ConstructTranscriptionModel<Client<T>> for TranscriptionModel<T>
where
    T: HttpClientExt + Clone + WasmCompatSend + 'static,
{
    fn construct(client: &Client<T>, model: String) -> Self {
        Self::new(client.clone(), model)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_mistral_transcription_response_deserialize() {
        let json = r#" {
          "model": "voxtral-mini-latest",
          "text": "The sun was setting slowly, casting long shadows across the empty field.",
          "language": null,
          "segments": [
            {
              "text": "The sun was setting slowly, casting long shadows across the empty field.",
              "start": 0.2,
              "end": 4.6,
              "speaker_id": "speaker_1",
              "type": "transcription_segment"
            }
          ],
          "usage": {
            "prompt_audio_seconds": 5,
            "prompt_tokens": 5,
            "total_tokens": 404,
            "completion_tokens": 24,
            "prompt_tokens_details": {
              "cached_tokens": 368
            }
          },
          "finish_reason": null
            }"#;

        let response: MistralTranscriptionResponse =
            serde_json::from_str(json).expect("should deserialize");

        assert_eq!(response.language, None);
        assert_eq!(response.model, VOXTRAL_MINI);
        assert_eq!(response.segments.len(), 1);

        let seg0 = &response.segments[0];
        assert_eq!(seg0.start, 0.2);
        assert_eq!(seg0.end, 4.6);
        assert_eq!(seg0.score, None);
        assert_eq!(seg0.speaker_id, Some("speaker_1".to_string()));
        assert_eq!(seg0.segment_type, "transcription_segment");

        assert_eq!(response.usage.prompt_audio_seconds, Some(5));
        assert_eq!(response.usage.prompt_tokens, 5);
        assert_eq!(response.usage.total_tokens, 404);
        let usage_token_details = response.usage.prompt_tokens_details.unwrap();
        let cached_token = usage_token_details.get("cached_tokens").unwrap();

        assert_eq!(cached_token.to_string().parse::<i32>().unwrap(), 368);
    }

    #[test]
    fn test_response_conversion() {
        let mistral_response = MistralTranscriptionResponse {
            language: Some("en".to_string()),
            model: VOXTRAL_MINI.to_string(),
            segments: vec![SegmentChunk {
                start: 0.0,
                end: 1.0,
                text: "Lorem Ipsum is simply dummy text of the printing and typesetting industry."
                    .into(),
                score: None,
                speaker_id: None,
                segment_type: "speech".to_string(),
            }],
            text: "Lorem Ipsum is simply dummy text of the printing and typesetting industry."
                .to_string(),
            usage: TranscriptionUsage {
                prompt_audio_seconds: Some(1),
                prompt_tokens: 10,
                total_tokens: 20,
                completion_tokens: 10,
                prompt_tokens_details: None,
            },
        };

        let response = mistral_response
            .normalize("mistral")
            .expect("conversion should succeed");

        assert_eq!(
            response.text,
            "Lorem Ipsum is simply dummy text of the printing and typesetting industry."
        );
        assert_eq!(response.provider, "mistral");
        assert_eq!(response.model.as_deref(), Some(VOXTRAL_MINI));
        assert_eq!(response.usage.input_tokens, 10);
        assert_eq!(response.usage.output_tokens, 10);
        assert_eq!(response.usage.total_tokens, 20);
    }

    #[tokio::test]
    async fn transcription_non_success_preserves_status_and_body() {
        use crate::client::transcription::TranscriptionClient;
        use crate::test_utils::RecordingHttpClient;
        use crate::transcription::{TranscriptionError, TranscriptionModel as _};

        let body = r#"{"error":{"message":"boom"}}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
        let client = Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.transcription_model(VOXTRAL_MINI);

        let error = match model
            .transcription_request()
            .data(vec![0u8; 16])
            .send()
            .await
        {
            Err(error) => error,
            Ok(_) => panic!("transcription should fail with non-success status"),
        };

        assert!(matches!(error, TranscriptionError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::SERVICE_UNAVAILABLE)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }
}
