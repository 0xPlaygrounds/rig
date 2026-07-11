use crate::audio_generation::{
    self, AudioGenerationError, AudioGenerationRequest, AudioGenerationResponse,
};
use crate::http_client::{self, HttpClientExt};
use crate::json_utils::merge_inplace;
use crate::providers::xai::client::{XAiExt, XAiRequestAuth};
use bytes::Bytes;
use serde_json::json;

// ================================================================
// xAI TTS API
// ================================================================
pub const TTS_1: &str = "tts-1";

#[derive(Clone)]
pub struct AudioGenerationModel<T = reqwest::Client, E = XAiExt> {
    client: crate::client::Client<E, T>,
    pub model: String,
}

impl<T, E> AudioGenerationModel<T, E> {
    pub(crate) fn new(client: crate::client::Client<E, T>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }
}

impl<T, E> audio_generation::AudioGenerationModel for AudioGenerationModel<T, E>
where
    T: HttpClientExt + Clone + std::fmt::Debug + Default + 'static,
    E: XAiRequestAuth
        + crate::client::DebugExt
        + crate::wasm_compat::WasmCompatSend
        + crate::wasm_compat::WasmCompatSync
        + Clone
        + 'static,
{
    type Response = Bytes;

    type Client = crate::client::Client<E, T>;

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
        let req = self.client.ext().authorize_request(req).await?;

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
    use super::*;
    use crate::audio_generation::AudioGenerationModel as _;
    use crate::client::audio_generation::AudioGenerationClient;

    #[tokio::test]
    async fn audio_generation_non_success_preserves_status_and_body() {
        use crate::test_utils::RecordingHttpClient;

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
