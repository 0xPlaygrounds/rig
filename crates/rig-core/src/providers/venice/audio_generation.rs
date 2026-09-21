//! Venice's text-to-speech model identifiers.
//!
//! ```
//! use rig_core::providers::venice::audio_generation::TTS_KOKORO;
//! assert_eq!(TTS_KOKORO, "tts-kokoro");
//! ```

/// Identifier for `tts-kokoro`.
pub const TTS_KOKORO: &str = "tts-kokoro";
/// `tts-xai-v1`
pub const TTS_XAI_V1: &str = "tts-xai-v1";
/// `tts-elevenlabs-turbo-v2-5`
pub const TTS_ELEVENLABS_TURBO_V2_5: &str = "tts-elevenlabs-turbo-v2-5";
/// `tts-inworld-1-5-max`
pub const TTS_INWORLD_1_5_MAX: &str = "tts-inworld-1-5-max";
