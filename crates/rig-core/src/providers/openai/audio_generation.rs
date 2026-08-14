use crate::providers::internal::audio_generation::{
    GenericAudioGenerationModel, RawAudioGenerationProvider,
};
use crate::providers::openai::{OpenAICompletionsExt, OpenAIResponsesExt};

pub const TTS_1: &str = "tts-1";
pub const TTS_1_HD: &str = "tts-1-hd";

/// OpenAI audio generation model.
pub type AudioGenerationModel<T = reqwest::Client> =
    GenericAudioGenerationModel<OpenAIResponsesExt, T>;

/// OpenAI audio generation model for a client using Chat Completions.
pub type CompletionsAudioGenerationModel<T = reqwest::Client> =
    GenericAudioGenerationModel<OpenAICompletionsExt, T>;

impl RawAudioGenerationProvider for OpenAIResponsesExt {
    const AUDIO_GENERATION_PATH: &'static str = "/audio/speech";
}

impl RawAudioGenerationProvider for OpenAICompletionsExt {
    const AUDIO_GENERATION_PATH: &'static str = "/audio/speech";
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_generation::{AudioGenerationError, AudioGenerationModel as _};
    use crate::client::audio_generation::AudioGenerationClient;
    use crate::providers::openai::Client;
    use crate::test_utils::RecordingHttpClient;

    #[tokio::test]
    async fn audio_generation_non_success_preserves_status_and_body() {
        let body = r#"{"error":{"message":"boom"}}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
        let client = Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.audio_generation_model(TTS_1);

        let request = model
            .audio_generation_request()
            .text("hello")
            .voice("alloy")
            .build();

        let error = model
            .audio_generation(request)
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
