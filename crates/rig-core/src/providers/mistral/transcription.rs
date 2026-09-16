//! Mistral's transcription model identifiers and its own view of a
//! transcript.
//!
//! The request runs on the shared OpenAI transcription wire, whose
//! [`MISTRAL`](crate::providers::openai::wire::MISTRAL) dialect carries the
//! `/v1/audio/transcriptions` path. What remains here is data: the model
//! identifiers, and [`MistralTranscriptionResponse`] — the typed read of
//! Mistral's own reply document, which models the diarization segments and
//! the audio-second accounting rig's normalized
//! [`TranscriptionResponse`](crate::transcription::TranscriptionResponse) has
//! no field for.

use serde::{Deserialize, Serialize};

// ================================================================
// Mistral Transcription API
// ================================================================

/// Voxtral Mini model (latest version)
pub const VOXTRAL_MINI: &str = "voxtral-mini-latest";
/// Voxtral Small model (latest version)
pub const VOXTRAL_SMALL: &str = "voxtral-small-latest";

/// Request usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionUsage {
    pub prompt_audio_seconds: Option<i32>,
    pub prompt_tokens: i32,
    pub total_tokens: i32,
    pub completion_tokens: i32,
    pub prompt_tokens_details: Option<serde_json::Value>,
}

impl std::fmt::Display for TranscriptionUsage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Usage:")?;
        writeln!(f, "  prompt_tokens:     {}", self.prompt_tokens)?;
        writeln!(f, "  completion_tokens: {}", self.completion_tokens)?;
        writeln!(f, "  total_tokens:      {}", self.total_tokens)?;
        if let Some(details) = &self.prompt_tokens_details {
            writeln!(f, "  prompt_token_details: {details:?}")?;
        } else {
            writeln!(f, "  prompt_token_details: N/A")?;
        }
        if let Some(secs) = self.prompt_audio_seconds {
            write!(f, "  audio_seconds:     {secs}")?;
        } else {
            write!(f, "  audio_seconds:     N/A")?;
        }
        Ok(())
    }
}

/// Diarization information, tells when each speaker started and ended talking plus what they said.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentChunk {
    /// Start time in seconds
    pub start: f32,
    /// End time in seconds
    pub end: f32,
    /// Segment transcribed text
    pub text: String,
    pub score: Option<f32>,
    /// Speaker identification.
    pub speaker_id: Option<String>,
    #[serde(rename = "type")]
    pub segment_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MistralTranscriptionResponse {
    /// Audio language
    pub language: Option<String>,
    /// Model name (e.g. voxtra-mini-latest)
    pub model: String,
    /// An array of transcript segments, each containing a portion of the transcribed text along with its start and end times in seconds and speaker id (if diarization was enabled).
    pub segments: Vec<SegmentChunk>,
    /// Audio Transcription
    pub text: String,
    /// Request token usage statistics
    pub usage: TranscriptionUsage,
}

#[cfg(test)]
mod tests;
