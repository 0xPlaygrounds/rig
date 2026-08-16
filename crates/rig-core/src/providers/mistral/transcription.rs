//! Implements Mistral (basic) transcription API
use serde::Deserialize;

use crate::http_client::HttpClientExt;
use crate::providers::internal::envelope::DirectPayload;
use crate::providers::internal::transcription::{
    TranscriptionFields, send_transcription, transcription_form,
};
use crate::providers::mistral::Client;
use crate::transcription::{self, TranscriptionError};
use crate::wasm_compat::WasmCompatSend;

// ================================================================
// Mistral Transcription API
// ================================================================

/// Voxtral Mini model (latest version) — Mistral's transcription model.
///
/// The live catalog reports `capabilities.audio_transcription: true` for this
/// id and no other Voxtral variant.
pub const VOXTRAL_MINI: &str = "voxtral-mini-latest";
/// Voxtral Small model (latest version).
///
/// **Not a transcription model**, despite living beside one: the live catalog
/// reports `audio_transcription: false` for it (and `audio: true`,
/// `completion_chat: true`), and `POST /v1/audio/transcriptions` answers
/// `400 Invalid model: voxtral-small-latest`. It is the audio-*chat* model —
/// pass it to `completion_model` with `input_audio` content. Use
/// [`VOXTRAL_MINI`] to transcribe.
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
        tracing::info!(target: "rig", "Mistral transcription token usage: {}", &value.usage);

        Ok(transcription::TranscriptionResponse {
            text: value.text.clone(),
            response: value,
        })
    }
}

pub type TranscriptionModel<T = reqwest::Client> =
    crate::providers::internal::transcription::GenericTranscriptionModel<
        crate::providers::mistral::client::MistralExt,
        T,
    >;

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
        mut request: transcription::TranscriptionRequest,
    ) -> Result<transcription::TranscriptionResponse<Self::Response>, TranscriptionError> {
        // Mistral's transcription endpoint has no `prompt` field; it has
        // always been dropped rather than sent.
        request.prompt = None;

        let body = transcription_form(
            request,
            TranscriptionFields {
                model: Some(&self.model),
            },
        )?;

        // Through the shared driver rather than a hand-rolled status split:
        // it takes the response apart with `into_parts`, so a failed call
        // keeps its headers and the caller can read `Retry-After` off the
        // error (rig#2210). Mistral was the only provider whose transcription
        // built that tail itself, and the tail it built dropped them.
        send_transcription::<_, DirectPayload<MistralTranscriptionResponse>>(
            &self.client,
            self.client.post("/v1/audio/transcriptions")?,
            body,
        )
        .await
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

    /// The outbound multipart form, which no cassette can pin: a multipart
    /// request is exported with *no* body (the proxy stores bodies as strings),
    /// so the recorded fixture matches on multipart-ness alone.
    ///
    /// The shared builder's field shape is covered in
    /// `providers/internal/transcription.rs`; what is Mistral's own is the
    /// endpoint, the `model` field, and that `prompt` — a field Mistral's
    /// transcription endpoint does not have — never reaches the wire.
    #[tokio::test]
    async fn transcription_form_carries_the_documented_fields() {
        use crate::client::transcription::TranscriptionClient;
        use crate::test_utils::RecordingHttpClient;
        use crate::transcription::TranscriptionModel as _;

        let body = r#"{"model":"voxtral-mini-latest","text":"hi","language":null,
            "segments":[],"usage":{"prompt_audio_seconds":1,"prompt_tokens":1,
            "total_tokens":2,"completion_tokens":1,"prompt_tokens_details":null}}"#;
        let http_client = RecordingHttpClient::new(body);
        let client = Client::builder()
            .api_key("test-key")
            .http_client(http_client.clone())
            .build()
            .expect("build client");

        client
            .transcription_model(VOXTRAL_MINI)
            .transcription_request()
            .data(vec![0u8; 8])
            .language("en".to_string())
            .prompt("this is not a Mistral field".to_string())
            .send()
            .await
            .expect("transcription should succeed");

        let request = http_client
            .requests()
            .pop()
            .expect("one request should be recorded");
        assert!(
            request.uri.ends_with("/v1/audio/transcriptions"),
            "Mistral's client base URL is the bare host, so the path carries its own v1: {}",
            request.uri
        );
        let form = String::from_utf8_lossy(&request.body).into_owned();
        for field in ["name=\"model\"", "name=\"file\"", "name=\"language\""] {
            assert!(form.contains(field), "{field} must ride the form: {form}");
        }
        assert!(
            form.contains(VOXTRAL_MINI),
            "the model must name the value it was built with: {form}"
        );
        assert!(
            !form.contains("name=\"prompt\""),
            "Mistral's transcription endpoint has no `prompt` field: {form}"
        );
    }

    /// A failed transcription keeps the response's headers, so a caller can
    /// read `Retry-After` and back off (rig#2210).
    ///
    /// The bundled reqwest client raises a non-2xx as a transport error that
    /// already carries its headers, so this is only reachable through a custom
    /// [`HttpClientExt`] that hands the response back — which is exactly the
    /// transport shape the shared drivers' own rig#2210 tests use
    /// (`providers/internal/transcription.rs`, `header_preservation_tests`).
    /// Mistral was the one provider whose transcription hand-rolled the
    /// status split instead of going through those drivers, and the hand-rolled
    /// tail dropped the headers on the floor.
    #[tokio::test]
    async fn transcription_non_success_preserves_response_headers() {
        use crate::client::transcription::TranscriptionClient;
        use crate::test_utils::RecordingHttpClient;
        use crate::transcription::TranscriptionModel as _;

        let mut headers = http::HeaderMap::new();
        headers.insert(
            http::header::RETRY_AFTER,
            http::HeaderValue::from_static("42"),
        );
        headers.insert(
            "mistral-correlation-id",
            http::HeaderValue::from_static("cid-1"),
        );

        let body = r#"{"message":"rate limited"}"#;
        let http_client = RecordingHttpClient::with_error_response_headers(
            http::StatusCode::TOO_MANY_REQUESTS,
            body,
            headers,
        );
        let client = Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.transcription_model(VOXTRAL_MINI);

        let error = model
            .transcription_request()
            .data(vec![0u8; 16])
            .send()
            .await
            .err()
            .expect("transcription should fail with non-success status");

        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::TOO_MANY_REQUESTS)
        );
        assert_eq!(error.provider_response_body(), Some(body));
        let preserved = error
            .provider_response_headers()
            .expect("a failed capability call preserves its response headers (rig#2210)");
        assert_eq!(
            preserved
                .get(http::header::RETRY_AFTER)
                .map(|value| value.as_bytes()),
            Some("42".as_bytes()),
            "`Retry-After` is the header the contract exists for"
        );
        assert!(
            preserved.contains_key("mistral-correlation-id"),
            "Mistral's transport id rides the same headers"
        );
    }
}
