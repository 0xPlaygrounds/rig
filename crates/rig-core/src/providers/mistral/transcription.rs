//! Mistral's transcription model identifiers.
//!
//! The request runs on the shared OpenAI transcription wire, whose
//! [`MISTRAL`](crate::providers::openai::wire::MISTRAL) dialect carries the
//! `/v1/audio/transcriptions` path and reads the reply. Mistral's own
//! fields — the diarization segments and the audio-second accounting — stay
//! readable on
//! [`TranscriptionResponse::raw`](crate::transcription::TranscriptionResponse::raw).

/// Voxtral Mini model (latest version)
pub const VOXTRAL_MINI: &str = "voxtral-mini-latest";
/// Voxtral Small model (latest version)
pub const VOXTRAL_SMALL: &str = "voxtral-small-latest";
