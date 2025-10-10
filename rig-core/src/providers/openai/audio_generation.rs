use crate::audio_generation::{
    self, AudioGenerationError, AudioGenerationRequest, AudioGenerationResponse,
};
use crate::http_client::{self, HttpClientExt};
use crate::providers::openai::Client;
use bytes::{Buf, Bytes};
use serde_json::json;

pub const TTS_1: &str = "tts-1";
pub const TTS_1_HD: &str = "tts-1-hd";

#[derive(Clone)]
pub struct AudioGenerationModel<T = reqwest::Client> {
    client: Client<T>,
    pub model: String,
}

impl<T> AudioGenerationModel<T> {
    pub fn new(client: Client<T>, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }
}

impl<T> audio_generation::AudioGenerationModel for AudioGenerationModel<T>
where
    T: HttpClientExt + Clone,
{
    type Response = Bytes;

    #[cfg_attr(feature = "worker", worker::send)]
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
            .header("Content-Type", "application/json")
            .body(body)
            .map_err(http_client::Error::from)?;

        let response = self.client.send(req).await?;

        if !response.status().is_success() {
            let status = response.status();
            let mut bytes: Bytes = response.into_body().await?;
            let mut as_slice = Vec::new();
            bytes.copy_to_slice(&mut as_slice);

            let text: String = String::from_utf8_lossy(&as_slice).into();

            return Err(AudioGenerationError::ProviderError(format!(
                "{}: {}",
                status, text
            )));
        }

        let bytes: Bytes = response.into_body().await?;

        Ok(AudioGenerationResponse {
            audio: bytes.to_vec(),
            response: bytes,
        })
    }
}
