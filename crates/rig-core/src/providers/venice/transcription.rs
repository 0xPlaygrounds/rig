//! Venice speech-to-text (`POST /audio/transcriptions`).
//!
//! Venice accepts the OpenAI multipart body (`file` plus `model`, `language`,
//! `prompt`, `temperature`) and answers with `{ "text": … }`, so the shared
//! OpenAI-style transcription model drives it unchanged.

use crate::http_client::HttpClientExt;
use crate::providers::internal::transcription::OpenAiTranscriptionClient;

use super::client::Client;

// ================================================================
// Venice Transcription API
// ================================================================
/// `openai/whisper-large-v3`
pub const WHISPER_LARGE_V3: &str = "openai/whisper-large-v3";
/// `nvidia/parakeet-tdt-0.6b-v3`
pub const PARAKEET_TDT_0_6B_V3: &str = "nvidia/parakeet-tdt-0.6b-v3";
/// `elevenlabs/scribe-v2`
pub const SCRIBE_V2: &str = "elevenlabs/scribe-v2";
/// `fal-ai/wizper`
pub const WIZPER: &str = "fal-ai/wizper";

/// Venice transcription model using the shared OpenAI-style implementation.
pub type TranscriptionModel<T = reqwest::Client> =
    crate::providers::internal::transcription::OpenAiTranscriptionModel<Client<T>>;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::transcription::TranscriptionClient;
    use crate::test_utils::RecordingHttpClient;
    use crate::transcription::TranscriptionModel as _;

    /// Venice's ASR model ids contain a `/` (`openai/whisper-large-v3`); the
    /// model must travel as a form field, never spliced into the URL path.
    #[tokio::test]
    async fn transcription_routes_model_in_multipart_body() {
        let http_client = RecordingHttpClient::new(r#"{"text":"Rig speaks","duration":1.6}"#);
        let client = crate::providers::venice::Client::builder()
            .api_key("test-key")
            .http_client(http_client.clone())
            .build()
            .expect("build client");
        let model = client.transcription_model(WHISPER_LARGE_V3);

        let response = model
            .transcription_request()
            .data(vec![1, 2, 3])
            .filename(Some("audio.mp3".to_owned()))
            .send()
            .await
            .expect("transcription should succeed");

        assert_eq!(response.text, "Rig speaks");

        let requests = http_client.requests();
        let recorded = requests.first().expect("one request");
        assert!(recorded.uri.ends_with("/audio/transcriptions"));
        let body = String::from_utf8_lossy(&recorded.body);
        assert!(body.contains(WHISPER_LARGE_V3));
    }
}
