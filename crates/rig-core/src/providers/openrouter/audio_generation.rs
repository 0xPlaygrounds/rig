use crate::audio_generation::{AudioGenerationError, AudioGenerationRequest};
use crate::providers::internal::audio_generation::{
    GenericAudioGenerationModel, RawAudioGenerationProvider,
};
use crate::providers::openrouter::OpenRouterExt;
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

/// OpenRouter audio generation model.
pub type AudioGenerationModel<T> = GenericAudioGenerationModel<OpenRouterExt, T>;

impl RawAudioGenerationProvider for OpenRouterExt {
    const AUDIO_GENERATION_PATH: &'static str = "/audio/speech";
    const PROVIDER_NAME: &'static str = "openrouter";
    const EXPLICIT_JSON_CONTENT_TYPE: bool = true;

    fn audio_generation_request_body(
        model: &str,
        request: AudioGenerationRequest,
    ) -> Result<serde_json::Value, AudioGenerationError> {
        let mut body_map: serde_json::Map<String, serde_json::Value> = [
            ("model".to_string(), json!(model)),
            ("input".to_string(), json!(request.text)),
            ("voice".to_string(), json!(request.voice)),
            ("response_format".to_string(), json!("mp3")),
            ("speed".to_string(), json!(request.speed)),
        ]
        .into_iter()
        .collect();

        if let Some(ref additional_params) = request.additional_params {
            let params = additional_params.as_object().ok_or_else(|| {
                AudioGenerationError::RequestError(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "additional audio generation parameters must be a JSON object",
                )))
            })?;
            for (k, v) in params {
                body_map.insert(k.clone(), v.clone());
            }
        }

        Ok(serde_json::Value::Object(body_map))
    }
}

#[cfg(test)]
mod tests;
