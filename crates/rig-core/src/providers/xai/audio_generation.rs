use crate::audio_generation::{
    self, AudioGenerationError, AudioGenerationRequest, AudioGenerationResponse,
};
use crate::http_client::{self, HttpClientExt};
use crate::json_utils::merge_inplace;
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

        let body = serde_json::to_vec(&body)?;

        let req = self
            .client
            .post("/v1/tts")?
            .body(body)
            .map_err(http_client::Error::from)?;

        let response = self.client.send(req).await?;

        if !response.status().is_success() {
            let status = response.status();
            let text = http_client::text(response).await?;

            return Err(AudioGenerationError::from_http_response(status, text));
        }

        let bytes: Bytes = response.into_body().await?.into();

        Ok(AudioGenerationResponse {
            audio: bytes.to_vec(),
            response: bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::audio_generation::{
        AudioGenerationError, AudioGenerationModel as _, AudioGenerationRequest,
    };
    use crate::client::audio_generation::AudioGenerationClient;
    use crate::test_utils::RecordingHttpClient;

    #[tokio::test]
    async fn audio_generation_non_success_response_preserves_status_and_body() {
        let body = r#"{"error":"invalid voice","code":"422"}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::UNPROCESSABLE_ENTITY, body);
        let client = crate::providers::xai::Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.audio_generation_model(super::TTS_1);

        let error = match model
            .audio_generation(AudioGenerationRequest {
                text: "hello".to_string(),
                voice: "eve".to_string(),
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
            Some(http::StatusCode::UNPROCESSABLE_ENTITY)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }
}
