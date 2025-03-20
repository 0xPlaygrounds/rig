use std::convert::Infallible;
use crate::audio_generation::{
    self, AudioGenerationError, AudioGenerationRequest, AudioGenerationResponse,
};
use crate::providers::openai::Client;
use serde_json::json;
use bytes::Bytes;

#[derive(Clone)]
pub struct AudioGenerationModel {
    client: Client,
    pub model: String,
}

impl AudioGenerationModel {
    pub fn new(client: Client, model: &str) -> Self {
        Self {
            client,
            model: model.to_string(),
        }
    }
}

impl TryFrom<Bytes> for AudioGenerationResponse<Bytes> {
    type Error = Infallible;

    fn try_from(value: Bytes) -> Result<Self, Self::Error> {
        Ok(Self {
            audio: value.to_vec(),
            response: value,
        })
    }
}

impl audio_generation::AudioGenerationModel for AudioGenerationModel {
    type Response = Bytes;

    async fn audio_generation(
        &self,
        request: AudioGenerationRequest,
    ) -> Result<AudioGenerationResponse<Self::Response>, AudioGenerationError> {
        let request = json!({
            "model": self.model,
            "input": request.text,
            "voice": request.voice,
            "speed": request.speed,
        });

        let response = self
            .client
            .post("/audio/speech")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(AudioGenerationError::ProviderError(format!(
                "{}: {}",
                response.status().to_string(),
                response.text().await?
            )));
        }
     
        response.bytes().await?.try_into()
    }
}
