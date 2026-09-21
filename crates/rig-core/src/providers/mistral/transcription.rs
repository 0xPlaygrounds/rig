//! Mistral's transcription model identifiers.
//!
//! Diarization and audio-second metadata remain in
//! [`crate::transcription::TranscriptionResponse::raw`].
//!
//! ```no_run
//! use rig_core::providers::{mistral, openai::wire::{MISTRAL, OpenAI}};
//! let wire = OpenAI::from_env_with(&MISTRAL)?.transcriptions(mistral::VOXTRAL_MINI);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

/// Voxtral Mini model (latest version)
pub const VOXTRAL_MINI: &str = "voxtral-mini-latest";
/// Voxtral Small model (latest version)
pub const VOXTRAL_SMALL: &str = "voxtral-small-latest";
