use crate::audio_generation::{AudioGenerationError, AudioGenerationRequest};
use crate::json_utils::merge_inplace;
use crate::providers::internal::audio_generation::{
    GenericAudioGenerationModel, RawAudioGenerationProvider,
};
use crate::providers::xai::client::XAi;
use serde_json::json;

// ================================================================
// xAI TTS API
// ================================================================
pub const TTS_1: &str = "tts-1";

/// xAI audio generation model.
pub type AudioGenerationModel<T = crate::http_client::BoxedHttpClient> =
    GenericAudioGenerationModel<XAi, T>;

impl RawAudioGenerationProvider for XAi {
    const AUDIO_GENERATION_PATH: &'static str = "/v1/tts";
    const PROVIDER_NAME: &'static str = "xai";

    fn audio_generation_request_body(
        _model: &str,
        request: AudioGenerationRequest,
    ) -> Result<serde_json::Value, AudioGenerationError> {
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

        Ok(body)
    }
}

#[cfg(test)]
mod tests;
