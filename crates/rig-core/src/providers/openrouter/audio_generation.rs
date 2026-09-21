//! OpenRouter's text-to-speech model identifiers.
//!
//! ```
//! use rig_core::providers::openrouter::audio_generation::KOKORO_82M;
//! assert_eq!(KOKORO_82M, "hexgrad/kokoro-82m");
//! ```

/// The `openai/gpt-4o-mini-tts-2025-12-15` model.
pub const GPT_4O_MINI_TTS: &str = "openai/gpt-4o-mini-tts-2025-12-15";
/// The `mistralai/voxtral-mini-tts-2603` model.
pub const VOXTRAL_MINI_TTS: &str = "mistralai/voxtral-mini-tts-2603";
/// The `hexgrad/kokoro-82m` model.
pub const KOKORO_82M: &str = "hexgrad/kokoro-82m";
