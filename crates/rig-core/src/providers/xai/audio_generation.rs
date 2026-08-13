use crate::audio_generation::{
    self, AudioGenerationError, AudioGenerationRequest, AudioGenerationResponse,
};
use crate::http_client::HttpClientExt;
use crate::json_utils::merge_inplace;
use crate::providers::internal::audio_generation::send_audio_generation;
use crate::providers::xai::Client;
use bytes::Bytes;
use serde_json::json;

// ================================================================
// xAI TTS API
// ================================================================
pub const TTS_1: &str = "tts-1";

#[derive(Clone)]
pub struct AudioGenerationModel<T = reqwest::Client> {
    client: Client<T>,
    pub model: String,
}

impl<T> AudioGenerationModel<T> {
    pub(crate) fn new(client: Client<T>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }
}

impl<T> audio_generation::AudioGenerationModel for AudioGenerationModel<T>
where
    T: HttpClientExt + Clone + std::fmt::Debug + Default + 'static,
{
    type Response = Bytes;

    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model)
    }

    async fn audio_generation(
        &self,
        request: AudioGenerationRequest,
    ) -> Result<AudioGenerationResponse<Self::Response>, AudioGenerationError> {
        let voice = if request.voice.is_empty() {
            "eve".to_string()
        } else {
            request.voice
        };

        let mut body = json!({
            "text": request.text,
            "voice_id": voice,
            "language": "en",
        });

        if let Some(additional_params) = request.additional_params {
            merge_inplace(&mut body, additional_params);
        }

        send_audio_generation(&self.client, self.client.post("/v1/tts")?, body).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_generation::AudioGenerationModel as _;
    use crate::client::audio_generation::AudioGenerationClient;
    use crate::test_utils::RecordingHttpClient;

    #[tokio::test]
    async fn shared_driver_keeps_xai_request_and_binary_response() {
        let http_client = RecordingHttpClient::new(Bytes::from_static(b"audio"));
        let client = crate::providers::xai::Client::builder()
            .api_key("test-key")
            .http_client(http_client.clone())
            .build()
            .expect("build client");
        let model = client.audio_generation_model(TTS_1);

        let response = model
            .audio_generation(
                model
                    .audio_generation_request()
                    .text("hello")
                    .voice("")
                    .build(),
            )
            .await
            .expect("audio generation should succeed");

        assert_eq!(response.audio, b"audio");
        let requests = http_client.requests();
        assert_eq!(requests[0].uri, "https://api.x.ai/v1/tts");
        let body: serde_json::Value =
            serde_json::from_slice(&requests[0].body).expect("request body should be JSON");
        assert_eq!(body["text"], "hello");
        assert_eq!(body["voice_id"], "eve");
        assert_eq!(body["language"], "en");
    }

    #[tokio::test]
    async fn audio_generation_non_success_preserves_status_and_body() {
        let body = r#"{"error":"boom","code":"503"}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
        let client = crate::providers::xai::Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.audio_generation_model(TTS_1);

        let request = model
            .audio_generation_request()
            .text("hello")
            .voice("eve")
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
