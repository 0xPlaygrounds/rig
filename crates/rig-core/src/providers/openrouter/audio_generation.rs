use crate::audio_generation::{
    self, AudioGenerationError, AudioGenerationRequest, AudioGenerationResponse,
};
use crate::http_client::{self, HttpClientExt};
use crate::providers::openrouter::Client;
use bytes::Bytes;
use serde_json::json;

// ================================================================
// Model constants
// ================================================================

/// The `openai/gpt-4o-mini-tts-2025-12-15` model.
pub const GPT_4O_MINI_TTS: &str = "openai/gpt-4o-mini-tts-2025-12-15";
/// The `mistralai/voxtral-mini-tts-2603` model.
pub const VOXTRAL_MINI_TTS: &str = "mistralai/voxtral-mini-tts-2603";
/// The `hexgrad/kokoro-82m` model.
pub const KOKORO_82M: &str = "hexgrad/kokoro-82m";

// ================================================================
// Model
// ================================================================

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
    T: HttpClientExt + Clone + std::fmt::Debug + Default + Send + 'static,
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
        let mut body_map: serde_json::Map<String, serde_json::Value> = [
            ("model".to_string(), json!(self.model)),
            ("input".to_string(), json!(request.text)),
            ("voice".to_string(), json!(request.voice)),
            ("response_format".to_string(), json!("mp3")),
            ("speed".to_string(), json!(request.speed)),
        ]
        .into_iter()
        .collect();

        if let Some(obj) = request
            .additional_params
            .as_ref()
            .and_then(|p| p.as_object())
        {
            for (k, v) in obj {
                body_map.insert(k.clone(), v.clone());
            }
        }

        let body = serde_json::to_vec(&serde_json::Value::Object(body_map))?;

        let req = self
            .client
            .post("/audio/speech")?
            .header("Content-Type", "application/json")
            .body(body)
            .map_err(http_client::Error::from)?;

        let response = self.client.send(req).await?;

        if !response.status().is_success() {
            let status = response.status();
            let text = http_client::text(response).await?;
            return Err(AudioGenerationError::ProviderError(format!(
                "{}: {}",
                status, text
            )));
        }

        let audio: Vec<u8> = response.into_body().await?;

        Ok(AudioGenerationResponse {
            audio: audio.clone(),
            response: Bytes::from(audio),
        })
    }
}
