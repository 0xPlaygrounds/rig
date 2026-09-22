//! OpenRouter's transcription model identifiers.
//!
//! OpenRouter's speech-to-text route takes a JSON body carrying the audio as
//! a base64 `input_audio` block rather than a multipart upload. The wire's
//! transcription decoder reads the reply; the gateway's own usage fields
//! (audio seconds, cost) stay readable on
//! [`TranscriptionResponse::raw`](crate::transcription::TranscriptionResponse::raw).
//!
//! ```no_run
//! use rig_core::providers::{openrouter, openai::wire::{OPENROUTER, OpenAI}};
//! let wire = OpenAI::from_env_with(&OPENROUTER)?.transcriptions(openrouter::WHISPER_1);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

/// The `openai/whisper-1` model.
pub const WHISPER_1: &str = "openai/whisper-1";
/// The `openai/whisper-large-v3-turbo` model.
pub const WHISPER_LARGE_V3_TURBO: &str = "openai/whisper-large-v3-turbo";
/// The `openai/whisper-large-v3` model.
pub const WHISPER_LARGE_V3: &str = "openai/whisper-large-v3";
/// The `openai/gpt-4o-transcribe` model.
pub const GPT_4O_TRANSCRIBE: &str = "openai/gpt-4o-transcribe";
/// The `openai/gpt-4o-mini-transcribe` model.
pub const GPT_4O_MINI_TRANSCRIBE: &str = "openai/gpt-4o-mini-transcribe";
/// The `google/chirp-3` model.
pub const CHIRP_3: &str = "google/chirp-3";
