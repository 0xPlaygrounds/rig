//! OpenRouter's transcription model identifiers and its own view of a
//! transcript.
//!
//! OpenRouter's speech-to-text route takes a JSON body carrying the audio as
//! a base64 `input_audio` block rather than a multipart upload, and answers
//! with the document [`TranscriptionResponse`] models — the text plus a usage
//! block that reports audio seconds and the gateway's cost, neither of which
//! rig's normalized
//! [`TranscriptionResponse`](crate::transcription::TranscriptionResponse) has
//! a field for.

use serde::{Deserialize, Serialize};

// ================================================================
// Model constants
// ================================================================

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

// ================================================================
// Request/Response types
// ================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResponse {
    pub text: String,
    #[serde(default)]
    pub usage: Option<TranscriptionUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionUsage {
    #[serde(default)]
    pub seconds: Option<f64>,
    #[serde(default)]
    pub total_tokens: Option<usize>,
    #[serde(default)]
    pub input_tokens: Option<usize>,
    #[serde(default)]
    pub output_tokens: Option<usize>,
    #[serde(default)]
    pub cost: Option<f64>,
}

#[cfg(test)]
mod tests;
