use crate::audio_generation::{
    self, AudioGenerationError, AudioGenerationRequest, AudioGenerationResponse,
};
use crate::providers::openai::Client;
use bytes::Bytes;
use serde_json::json;

pub const TTS_1: &str = "tts-1";
pub const TTS_1_HD: &str = "tts-1-hd";

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
                response.status(),
                response.text().await?
            )));
        }

        let bytes = response.bytes().await?;

        Ok(AudioGenerationResponse {
            audio: bytes.to_vec(),
            response: bytes,
        })
    }
}
