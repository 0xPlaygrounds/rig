//! Venice text-to-speech (`POST /audio/speech`).

use serde_json::json;

use crate::audio_generation::{AudioGenerationError, AudioGenerationRequest};
use crate::json_utils::merge_inplace;
use crate::providers::internal::audio_generation::{
    GenericAudioGenerationModel, RawAudioGenerationProvider,
};
use crate::providers::venice::VeniceExt;

// ================================================================
// Venice TTS API
// ================================================================
/// `tts-kokoro` — Venice's default TTS model.
pub const TTS_KOKORO: &str = "tts-kokoro";
/// `tts-xai-v1`
pub const TTS_XAI_V1: &str = "tts-xai-v1";
/// `tts-elevenlabs-turbo-v2-5`
pub const TTS_ELEVENLABS_TURBO_V2_5: &str = "tts-elevenlabs-turbo-v2-5";
/// `tts-inworld-1-5-max`
pub const TTS_INWORLD_1_5_MAX: &str = "tts-inworld-1-5-max";

/// Kokoro's default voice, used when a request carries no voice.
const DEFAULT_VOICE: &str = "af_sky";

/// Venice audio generation model.
pub type AudioGenerationModel<T> = GenericAudioGenerationModel<VeniceExt, T>;

impl RawAudioGenerationProvider for VeniceExt {
    const AUDIO_GENERATION_PATH: &'static str = "/audio/speech";
    const PROVIDER_NAME: &'static str = "venice";

    fn audio_generation_request_body(
        model: &str,
        request: AudioGenerationRequest,
    ) -> Result<serde_json::Value, AudioGenerationError> {
        // Venice validates `voice` against the model's own voice list and
        // rejects an empty string, so an unset voice falls back to the
        // documented default rather than being sent blank.
        let voice = if request.voice.is_empty() {
            DEFAULT_VOICE.to_string()
        } else {
            request.voice
        };

        let mut body = json!({
            "model": model,
            "input": request.text,
            "voice": voice,
            "speed": request.speed,
        });

        if let Some(additional_params) = request.additional_params {
            merge_inplace(&mut body, additional_params);
        }

        Ok(body)
    }
}

#[cfg(test)]
mod tests;
