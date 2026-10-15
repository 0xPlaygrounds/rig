//! Mistral's transcription model identifiers.
//!
//! `timestamp_granularities` (`["segment"]` or `["word"]`) in
//! `additional_params` fills
//! [`TranscriptionResponse::segments`](crate::transcription::TranscriptionResponse::segments),
//! one per word at word granularity. `diarize: true` with `["segment"]` adds
//! speaker labels such as `"speaker_1"`. Audio-second usage remains in
//! [`TranscriptionResponse::raw`](crate::transcription::TranscriptionResponse::raw).
//!
//! ```no_run
//! use rig_core::providers::mistral;
//! let model = mistral::from_env()?.transcription(mistral::VOXTRAL_MINI);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

/// Voxtral Mini model (latest version)
pub const VOXTRAL_MINI: &str = "voxtral-mini-latest";
/// Voxtral Small model (latest version)
pub const VOXTRAL_SMALL: &str = "voxtral-small-latest";
