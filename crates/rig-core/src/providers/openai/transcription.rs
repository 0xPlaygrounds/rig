use crate::http_client::HttpClientExt;
use crate::providers::internal::transcription::OpenAiTranscriptionClient;
use crate::providers::openai::{Client, CompletionsClient};
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
    /// What the transcription cost, as the endpoint reported it.
    ///
    /// Both live model families return this beside the transcript, and it is
    /// the only accounting a caller gets for a transcription — the normalized
    /// [`transcription::TranscriptionResponse`] carries no usage slot, so
    /// dropping it here dropped it everywhere. Optional because a compatible
    /// provider on this wire may not report one.
    #[serde(default)]
    pub usage: Option<TranscriptionUsage>,
}

/// The accounting an OpenAI-style transcription endpoint reports.
///
/// Two shapes are live and they bill differently: `whisper-1` bills by audio
/// duration (`{"type":"duration","seconds":6}`), while the
/// `gpt-4o-transcribe` family bills by token
/// (`{"type":"tokens","input_tokens":54,…}`).
///
/// Each modeled variant pins the wire's own `type`, so selection cannot turn
/// on which optional keys a payload happens to carry: a future shape that
/// reported `seconds` *and* token counts would otherwise decode as a duration
/// and silently drop every token count. Anything whose `type` is unmodeled
/// falls to the verbatim catch-all rather than failing the whole
/// transcription — the same invariant the Responses `Output` enum keeps for
/// unmodeled output items.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum TranscriptionUsage {
    /// Duration-billed models.
    Duration {
        /// Always `"duration"`; pins this variant to its wire tag.
        r#type: DurationTag,
        /// Length of the audio, in seconds.
        seconds: f64,
    },
    /// Token-billed models.
    Tokens {
        /// Always `"tokens"`; pins this variant to its wire tag.
        r#type: TokensTag,
        /// Tokens consumed by the audio and any prompt.
        input_tokens: u64,
        /// How the input tokens split between audio and text, when the
        /// provider breaks it down. The two are billed at different rates, so
        /// `input_tokens` alone does not determine what a turn cost.
        #[serde(default)]
        input_token_details: Option<TranscriptionInputTokenDetails>,
        /// Tokens in the transcript.
        output_tokens: u64,
        /// `input_tokens + output_tokens`, as the provider reported it.
        total_tokens: u64,
    },
    /// A shape this version does not model, preserved as sent.
    Other(serde_json::Value),
}

/// The wire tag of [`TranscriptionUsage::Duration`].
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DurationTag {
    /// `"duration"`.
    Duration,
}

/// The wire tag of [`TranscriptionUsage::Tokens`].
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TokensTag {
    /// `"tokens"`.
    Tokens,
}

/// How a token-billed transcription's input tokens split by modality.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
pub struct TranscriptionInputTokenDetails {
    /// Input tokens attributable to the audio.
    #[serde(default)]
    pub audio_tokens: u64,
    /// Input tokens attributable to text (a prompt, for instance).
    #[serde(default)]
    pub text_tokens: u64,
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

/// OpenAI transcription model using the shared OpenAI-style implementation.
pub type TranscriptionModel<T = reqwest::Client> =
    crate::providers::internal::transcription::OpenAiTranscriptionModel<Client<T>>;

/// OpenAI transcription model for a client using Chat Completions.
pub type CompletionsTranscriptionModel<T = reqwest::Client> =
    crate::providers::internal::transcription::OpenAiTranscriptionModel<CompletionsClient<T>>;

impl<T> OpenAiTranscriptionClient for Client<T>
where
    T: HttpClientExt + Clone + 'static,
{
    const MODEL_IN_FORM: bool = true;

    fn transcription_request(
        &self,
        _model: &str,
    ) -> crate::http_client::Result<crate::http_client::Builder> {
        self.post("/audio/transcriptions")
    }
}

impl<T> OpenAiTranscriptionClient for CompletionsClient<T>
where
    T: HttpClientExt + Clone + 'static,
{
    const MODEL_IN_FORM: bool = true;

    fn transcription_request(
        &self,
        _model: &str,
    ) -> crate::http_client::Result<crate::http_client::Builder> {
        self.post("/audio/transcriptions")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::transcription::TranscriptionClient;
    use crate::test_utils::RecordingHttpClient;
    use crate::transcription::TranscriptionModel as _;

    #[tokio::test]
    async fn transcription_routes_model_in_multipart_body() {
        let http_client = RecordingHttpClient::new(r#"{"text":"transcribed"}"#);
        let client = Client::builder()
            .api_key("test-key")
            .http_client(http_client.clone())
            .build()
            .expect("build client");
        let model = client.transcription_model(WHISPER_1);

        let response = model
            .transcription_request()
            .data(vec![1, 2, 3])
            .filename(Some("audio.mp3".to_owned()))
            .send()
            .await
            .expect("transcription should succeed");

        assert_eq!(response.text, "transcribed");
        let request = http_client
            .requests()
            .into_iter()
            .next()
            .expect("request should be captured");
        assert_eq!(
            request.uri,
            "https://api.openai.com/v1/audio/transcriptions"
        );
        let body = String::from_utf8_lossy(&request.body);
        assert!(
            body.contains("name=\"model\"\r\n\r\nwhisper-1\r\n"),
            "{body}"
        );
        assert!(
            body.contains("name=\"file\"; filename=\"audio.mp3\""),
            "{body}"
        );
    }

    /// The two live shapes, and the catch-all that keeps a third from failing
    /// the transcription. Recorded turns of the first two are replayed in
    /// `transcription_usage_matrix`; this pins the decode itself, including
    /// the shapes no live model produces.
    #[test]
    fn usage_decodes_both_billing_shapes_and_keeps_unknown_ones() {
        fn usage(body: &str) -> Option<TranscriptionUsage> {
            serde_json::from_str::<TranscriptionResponse>(body)
                .expect("response should decode")
                .usage
        }

        assert_eq!(
            usage(r#"{"text":"hi","usage":{"type":"duration","seconds":6}}"#),
            Some(TranscriptionUsage::Duration {
                r#type: DurationTag::Duration,
                seconds: 6.0
            })
        );
        assert_eq!(
            usage(
                r#"{"text":"hi","usage":{"type":"tokens","input_tokens":54,
                   "input_token_details":{"audio_tokens":54,"text_tokens":0},
                   "output_tokens":16,"total_tokens":70}}"#
            ),
            Some(TranscriptionUsage::Tokens {
                r#type: TokensTag::Tokens,
                input_tokens: 54,
                input_token_details: Some(TranscriptionInputTokenDetails {
                    audio_tokens: 54,
                    text_tokens: 0,
                }),
                output_tokens: 16,
                total_tokens: 70,
            })
        );
        // The breakdown is optional: a provider that omits it still decodes as
        // a token-billed turn rather than falling to the catch-all.
        assert_eq!(
            usage(
                r#"{"text":"hi","usage":{"type":"tokens","input_tokens":54,
                   "output_tokens":16,"total_tokens":70}}"#
            ),
            Some(TranscriptionUsage::Tokens {
                r#type: TokensTag::Tokens,
                input_tokens: 54,
                input_token_details: None,
                output_tokens: 16,
                total_tokens: 70,
            })
        );
        // The tag decides, not which optional keys are present: a token-billed
        // payload that also reported `seconds` must not decode as a duration
        // and drop every token count.
        assert!(matches!(
            usage(
                r#"{"text":"hi","usage":{"type":"tokens","seconds":6,"input_tokens":54,
                   "output_tokens":16,"total_tokens":70}}"#
            ),
            Some(TranscriptionUsage::Tokens {
                total_tokens: 70,
                ..
            })
        ));
        assert!(matches!(
            usage(r#"{"text":"hi","usage":{"type":"credits","spent":3}}"#),
            Some(TranscriptionUsage::Other(_))
        ));
        // A token-shaped payload missing a required total degrades to the
        // catch-all rather than failing the transcription.
        assert!(matches!(
            usage(r#"{"text":"hi","usage":{"type":"tokens","input_tokens":54}}"#),
            Some(TranscriptionUsage::Other(_))
        ));
        assert_eq!(usage(r#"{"text":"hi"}"#), None);
        assert_eq!(usage(r#"{"text":"hi","usage":null}"#), None);
    }

    #[tokio::test]
    async fn transcription_http_non_success_preserves_status_and_body() {
        let body = r#"{"error":{"message":"bad audio","type":"invalid_request_error"}}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::BAD_REQUEST, body);
        let client = Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.transcription_model(WHISPER_1);

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
            Some(http::StatusCode::BAD_REQUEST)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }
}
