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
        let body = build_audio_generation_body(&request)?;

        let req = self
            .client
            .post("/v1/tts")?
            .body(body)
            .map_err(http_client::Error::from)?;

        let response = self.client.send(req).await?;

        let status = response.status();
        let bytes: Bytes = response.into_body().await?;
        parse_audio_generation_response(status, bytes.to_vec())
    }
}

/// Build the serialized TTS request body. Pure.
///
/// xAI's TTS route has no model field; an empty voice falls back to `eve`.
pub(crate) fn build_audio_generation_body(
    request: &AudioGenerationRequest,
) -> Result<Vec<u8>, AudioGenerationError> {
    let voice = if request.voice.is_empty() {
        "eve"
    } else {
        request.voice.as_str()
    };

    let mut body = json!({
        "text": request.text,
        "voice_id": voice,
        "language": "en",
    });

    if let Some(additional_params) = request.additional_params.clone() {
        merge_inplace(&mut body, additional_params);
    }

    Ok(serde_json::to_vec(&body)?)
}

/// Parse a TTS response: success bodies are raw audio bytes. Pure.
pub(crate) fn parse_audio_generation_response(
    status: http::StatusCode,
    body: Vec<u8>,
) -> Result<AudioGenerationResponse<Bytes>, AudioGenerationError> {
    if !status.is_success() {
        return Err(AudioGenerationError::from_http_response(
            status,
            String::from_utf8_lossy(&body).into_owned(),
        ));
    }
    Ok(AudioGenerationResponse {
        audio: body.clone(),
        response: Bytes::from(body),
    })
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

        let request = AudioGenerationRequest::new("hello", "eve");

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
