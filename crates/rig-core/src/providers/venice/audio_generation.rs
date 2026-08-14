//! Venice text-to-speech (`POST /audio/speech`).

use serde_json::json;

use crate::audio_generation::{AudioGenerationError, AudioGenerationRequest};
use crate::json_utils::merge_inplace;
use crate::providers::internal::audio_generation::{
    GenericAudioGenerationModel, RawAudioGenerationProvider,
};
use crate::providers::venice::VeniceExt;

// ================================================================
// Venice TTS API
// ================================================================
/// `tts-kokoro` — Venice's default TTS model.
pub const TTS_KOKORO: &str = "tts-kokoro";
/// `tts-xai-v1`
pub const TTS_XAI_V1: &str = "tts-xai-v1";
/// `tts-elevenlabs-turbo-v2-5`
pub const TTS_ELEVENLABS_TURBO_V2_5: &str = "tts-elevenlabs-turbo-v2-5";
/// `tts-inworld-1-5-max`
pub const TTS_INWORLD_1_5_MAX: &str = "tts-inworld-1-5-max";

/// Kokoro's default voice, used when a request carries no voice.
const DEFAULT_VOICE: &str = "af_sky";

/// Venice audio generation model.
pub type AudioGenerationModel<T = reqwest::Client> = GenericAudioGenerationModel<VeniceExt, T>;

impl RawAudioGenerationProvider for VeniceExt {
    const AUDIO_GENERATION_PATH: &'static str = "/audio/speech";

    fn audio_generation_request_body(
        model: &str,
        request: AudioGenerationRequest,
    ) -> Result<serde_json::Value, AudioGenerationError> {
        // Venice validates `voice` against the model's own voice list and
        // rejects an empty string, so an unset voice falls back to the
        // documented default rather than being sent blank.
        let voice = if request.voice.is_empty() {
            DEFAULT_VOICE.to_string()
        } else {
            request.voice
        };

        let mut body = json!({
            "model": model,
            "input": request.text,
            "voice": voice,
            "speed": request.speed,
        });

        if let Some(additional_params) = request.additional_params {
            merge_inplace(&mut body, additional_params);
        }

        Ok(body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_generation::AudioGenerationModel as _;
    use crate::client::audio_generation::AudioGenerationClient;
    use crate::test_utils::RecordingHttpClient;

    #[tokio::test]
    async fn audio_generation_non_success_preserves_status_and_body() {
        let body = r#"{"error":"Insufficient USD or DIEM balance"}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::PAYMENT_REQUIRED, body);
        let client = crate::providers::venice::Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.audio_generation_model(TTS_KOKORO);

        let request = model
            .audio_generation_request()
            .text("hello")
            .voice("af_sky")
            .build();

        let error = model
            .audio_generation(request)
            .await
            .err()
            .expect("should fail with non-success status");

        assert!(matches!(error, AudioGenerationError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::PAYMENT_REQUIRED)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }

    /// An unset voice must not reach Venice as `""`, which it rejects.
    #[tokio::test]
    async fn empty_voice_falls_back_to_the_model_default() {
        let http_client = RecordingHttpClient::new("audio-bytes");
        let client = crate::providers::venice::Client::builder()
            .api_key("test-key")
            .http_client(http_client.clone())
            .build()
            .expect("build client");
        let model = client.audio_generation_model(TTS_KOKORO);

        let request = model
            .audio_generation_request()
            .text("hello")
            .voice("")
            .build();
        model
            .audio_generation(request)
            .await
            .expect("audio generation should succeed");

        let requests = http_client.requests();
        let recorded = requests.first().expect("one request");
        assert!(recorded.uri.ends_with("/audio/speech"));
        let body: serde_json::Value =
            serde_json::from_slice(&recorded.body).expect("body should be JSON");
        assert_eq!(body["voice"], DEFAULT_VOICE);
        assert_eq!(body["model"], TTS_KOKORO);
        assert_eq!(body["input"], "hello");
    }
}
