//! Implements Mistral (basic) transcription API
use bytes::Bytes;
use serde::Deserialize;

use crate::http_client::multipart::Part;
use crate::http_client::{HttpClientExt, MultipartForm};
use crate::providers::mistral::Client;
use crate::transcription::{self, TranscriptionError};
use crate::wasm_compat::WasmCompatSend;

// ================================================================
// Mistral Transcription API
// ================================================================

/// Voxtral Mini model (latest version)
pub const VOXTRAL_MINI: &str = "voxtral-mini-latest";
/// Voxtral Small model (latest version)
pub const VOXTRAL_SMALL: &str = "voxtral-small-latest";

/// Request usage statistics
#[derive(Debug, Deserialize)]
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
#[derive(Debug, Deserialize)]
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

#[derive(Debug, Deserialize)]
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

impl TryFrom<MistralTranscriptionResponse>
    for transcription::TranscriptionResponse<MistralTranscriptionResponse>
{
    type Error = TranscriptionError;

    fn try_from(value: MistralTranscriptionResponse) -> Result<Self, Self::Error> {
        Ok(transcription::TranscriptionResponse {
            text: value.text.clone(),
            response: value,
        })
    }
}

#[derive(Clone)]
pub struct TranscriptionModel<T = reqwest::Client> {
    client: Client<T>,
    pub model: String,
}

impl<T> transcription::TranscriptionModel for TranscriptionModel<T>
where
    T: HttpClientExt + Clone + std::fmt::Debug + Default + WasmCompatSend + 'static,
{
    type Response = MistralTranscriptionResponse;
    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model)
    }

    async fn transcription(
        &self,
        request: transcription::TranscriptionRequest,
    ) -> Result<transcription::TranscriptionResponse<Self::Response>, TranscriptionError> {
        let data = request.data;

        let mut body = MultipartForm::new()
            .text("model", self.model.clone())
            .part(Part::bytes("file", data).filename(request.filename.clone()));

        if let Some(language) = request.language {
            body = body.text("language", language);
        }

        if let Some(ref temperature) = request.temperature {
            body = body.text("temperature", temperature.to_string());
        }

        if let Some(ref additional_params) = request.additional_params {
            for (key, value) in additional_params.as_object().ok_or_else(|| {
                TranscriptionError::RequestError(
                    "Additional Parameters to Mistral Transcription should be a map".into(),
                )
            })? {
                body = body.text(key.to_owned(), value.to_string());
            }
        }

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

        if response.status().is_success() {
            let response_bytes = response.into_body().await?;
            let response_body: MistralTranscriptionResponse =
                serde_json::from_slice(&response_bytes)?;

            tracing::info!(target: "rig", "Mistral transcription token usage: {}", &response_body.usage);

            Ok(transcription::TranscriptionResponse::try_from(
                response_body,
            )?)
        } else {
            let text = String::from_utf8_lossy(&response.into_body().await?).into();
            Err(TranscriptionError::ProviderError(text))
        }
    }
}

impl<T> TranscriptionModel<T> {
    pub fn new(client: Client<T>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::transcription::TranscriptionResponse;

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

        let response: TranscriptionResponse<MistralTranscriptionResponse> = mistral_response
            .try_into()
            .expect("conversion should succeed");

        assert_eq!(
            response.text,
            "Lorem Ipsum is simply dummy text of the printing and typesetting industry."
        );
        assert_eq!(response.response.model, VOXTRAL_MINI);
        assert_eq!(response.response.language, Some("en".to_string()));
    }
}
