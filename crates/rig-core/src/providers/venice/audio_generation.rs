//! Venice's text-to-speech model identifiers.
//!
//! The request runs on the shared OpenAI speech wire, whose
//! [`VENICE`](crate::providers::openai::wire::VENICE) dialect carries the
//! `/audio/speech` path.

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
