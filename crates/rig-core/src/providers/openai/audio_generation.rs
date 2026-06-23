use crate::audio_generation::{
    self, AudioGenerationError, AudioGenerationRequest, AudioGenerationResponse,
};
use crate::http_client::{self, HttpClientExt};
use crate::providers::openai::Client;
use bytes::Bytes;
use serde_json::json;

pub const TTS_1: &str = "tts-1";
pub const TTS_1_HD: &str = "tts-1-hd";

#[derive(Clone)]
pub struct AudioGenerationModel<T = reqwest::Client> {
    client: Client<T>,
    pub model: String,
}

impl<T> AudioGenerationModel<T> {
    pub fn new(client: Client<T>, model: impl Into<String>) -> Self {
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
        let body = serde_json::to_vec(&json!({
            "model": self.model,
            "input": request.text,
            "voice": request.voice,
            "speed": request.speed,
        }))?;

        let req = self
            .client
            .post("/audio/speech")?
            .body(body)
            .map_err(http_client::Error::from)?;

        let response = self.client.send(req).await?;

        if !response.status().is_success() {
            let status = response.status();
            let bytes: Bytes = response.into_body().await?;

            return Err(AudioGenerationError::from_http_response(
                status,
                String::from_utf8_lossy(&bytes),
            ));
        }

        let bytes: Bytes = response.into_body().await?;

        Ok(AudioGenerationResponse {
            audio: bytes.to_vec(),
            response: bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_generation::AudioGenerationModel as _;
    use crate::test_utils::RecordingHttpClient;

    #[tokio::test]
    async fn audio_generation_http_non_success_preserves_status_and_body() {
        let body = r#"{"error":{"message":"invalid voice","type":"invalid_request_error"}}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
        let client = Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = AudioGenerationModel::new(client, TTS_1);

        let error = match model
            .audio_generation(AudioGenerationRequest {
                text: "hello".to_string(),
                voice: "alloy".to_string(),
                speed: 1.0,
                additional_params: None,
            })
            .await
        {
            Err(error) => error,
            Ok(_) => panic!("audio generation should fail with non-success status"),
        };

        assert!(matches!(error, AudioGenerationError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::SERVICE_UNAVAILABLE)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }
}
