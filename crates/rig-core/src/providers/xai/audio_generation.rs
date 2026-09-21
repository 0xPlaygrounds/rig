//! xAI's text-to-speech model identifier.
//!
//! The speech encoder omits the model field and uses `voice_id`, defaulting to `eve`.
//!
//! ```
//! use rig_core::providers::xai::audio_generation::TTS_1;
//! assert_eq!(TTS_1, "tts-1");
//! ```

pub const TTS_1: &str = "tts-1";
