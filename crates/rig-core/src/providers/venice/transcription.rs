//! Venice's transcription model identifiers.
//!
//! ```
//! use rig_core::providers::venice::transcription::WHISPER_LARGE_V3;
//! assert_eq!(WHISPER_LARGE_V3, "openai/whisper-large-v3");
//! ```

/// `openai/whisper-large-v3`
pub const WHISPER_LARGE_V3: &str = "openai/whisper-large-v3";
/// `nvidia/parakeet-tdt-0.6b-v3`
pub const PARAKEET_TDT_0_6B_V3: &str = "nvidia/parakeet-tdt-0.6b-v3";
/// `elevenlabs/scribe-v2`
pub const SCRIBE_V2: &str = "elevenlabs/scribe-v2";
/// `fal-ai/wizper`
pub const WIZPER: &str = "fal-ai/wizper";
