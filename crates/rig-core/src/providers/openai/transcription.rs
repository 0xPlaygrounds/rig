//! The transcription reply shared by OpenAI and the endpoints compatible
//! with it, and OpenAI's transcription model identifier.
//!
//! `whisper-1` returns timed segments with `response_format: "verbose_json"`,
//! and timed words as well with `timestamp_granularities: ["segment", "word"]`.
//! `gpt-4o-transcribe-diarize` labels segments with speakers such as `"A"`
//! under `response_format: "diarized_json"`.
use crate::completion::Usage;
use crate::error::ProviderError;
use crate::json_utils;
use crate::transcription::{self, TranscriptSegment, TranscriptWord};
use serde::{Deserialize, Serialize};

pub const WHISPER_1: &str = "whisper-1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResponse {
    pub text: String,
    /// Optional endpoint-reported usage. Token counts normalize into
    /// [`transcription::TranscriptionResponse::usage`]; duration remains raw.
    #[serde(default)]
    pub usage: Option<TranscriptionUsage>,
    /// Language of the audio, reported by verbose and diarized formats.
    #[serde(default)]
    pub language: Option<String>,
    /// Length of the audio in seconds, reported by verbose and diarized
    /// formats.
    #[serde(default)]
    pub duration: Option<f64>,
    /// Confidence in the whole transcript, from 0 to 1, reported by some
    /// verbose formats.
    #[serde(default)]
    pub confidence: Option<f64>,
    /// Timed segments, reported by verbose and diarized formats.
    #[serde(default, deserialize_with = "json_utils::null_or_default")]
    pub segments: Vec<TranscriptionSegment>,
    /// Timed words, reported when word timestamps were requested.
    #[serde(default, deserialize_with = "json_utils::null_or_default")]
    pub words: Vec<TranscriptionWord>,
}

/// A timed segment. Fields beyond these, such as Whisper's token ids and
/// log probabilities, stay on the raw document.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TranscriptionSegment {
    /// Start, in seconds. Mistral's schema allows no time.
    #[serde(default)]
    pub start: Option<f64>,
    /// End, in seconds. Mistral's schema allows no time.
    #[serde(default)]
    pub end: Option<f64>,
    /// The segment's text.
    pub text: String,
    /// The diarized speaker. Mistral names this field `speaker_id`.
    #[serde(default, alias = "speaker_id")]
    pub speaker: Option<Speaker>,
}

/// A timed word.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TranscriptionWord {
    /// The word.
    pub word: String,
    /// Start, in seconds.
    pub start: f64,
    /// End, in seconds.
    pub end: f64,
    /// The diarized speaker.
    #[serde(default)]
    pub speaker: Option<Speaker>,
    /// Confidence from 0 to 1.
    #[serde(default)]
    pub confidence: Option<f64>,
}

/// A speaker label: an index on some gateways, a name on others.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum Speaker {
    /// A numeric speaker index.
    Index(u64),
    /// A speaker name or letter.
    Label(String),
}

impl From<Speaker> for String {
    fn from(speaker: Speaker) -> Self {
        match speaker {
            Speaker::Index(index) => index.to_string(),
            Speaker::Label(label) => label,
        }
    }
}

/// Duration or token accounting selected by the wire's `type` tag.
/// Payloads that do not match a modeled shape are preserved verbatim in `Other`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum TranscriptionUsage {
    /// Duration-billed models.
    Duration {
        /// Always `"duration"`; pins this variant to its wire tag.
        r#type: DurationTag,
        /// Length of the audio, in seconds.
        seconds: f64,
    },
    /// Token-billed models.
    Tokens {
        /// Always `"tokens"`; pins this variant to its wire tag.
        r#type: TokensTag,
        /// Tokens consumed by the audio and any prompt.
        input_tokens: u64,
        /// How the input tokens split between audio and text, when the
        /// provider breaks it down. The two are billed at different rates, so
        /// `input_tokens` alone does not determine what a turn cost.
        #[serde(default)]
        input_token_details: Option<TranscriptionInputTokenDetails>,
        /// Tokens in the transcript.
        output_tokens: u64,
        /// `input_tokens + output_tokens`, as the provider reported it.
        total_tokens: u64,
    },
    /// A shape this version does not model, preserved as sent.
    Other(serde_json::Value),
}

/// The wire tag of [`TranscriptionUsage::Duration`].
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DurationTag {
    /// `"duration"`.
    Duration,
}

/// The wire tag of [`TranscriptionUsage::Tokens`].
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TokensTag {
    /// `"tokens"`.
    Tokens,
}

/// How a token-billed transcription's input tokens split by modality.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct TranscriptionInputTokenDetails {
    /// Input tokens attributable to the audio.
    #[serde(default)]
    pub audio_tokens: u64,
    /// Input tokens attributable to text (a prompt, for instance).
    #[serde(default)]
    pub text_tokens: u64,
}

impl TranscriptionResponse {
    /// Convert this payload into Rig's transcription response. The driver
    /// writes the provider, request id and `raw` afterwards.
    pub fn normalize(self) -> Result<transcription::TranscriptionResponse, ProviderError> {
        let usage = match &self.usage {
            Some(TranscriptionUsage::Tokens {
                input_tokens,
                output_tokens,
                total_tokens,
                ..
            }) => Usage {
                input_tokens: Some(*input_tokens),
                output_tokens: Some(*output_tokens),
                total_tokens: Some(*total_tokens),
                ..Default::default()
            },
            // Duration billing reports no token counts; the seconds stay
            // reachable on the raw payload.
            Some(TranscriptionUsage::Duration { .. })
            | Some(TranscriptionUsage::Other(_))
            | None => Usage::default(),
        };
        // A segment without both times is not a timed segment; its text is
        // still in `text`.
        let segments = self
            .segments
            .into_iter()
            .filter_map(|segment| {
                Some(TranscriptSegment {
                    start: segment.start?,
                    end: segment.end?,
                    text: segment.text,
                    speaker: segment.speaker.map(String::from),
                })
            })
            .collect();
        let words = self
            .words
            .into_iter()
            .map(|word| TranscriptWord {
                word: word.word,
                start: word.start,
                end: word.end,
                speaker: word.speaker.map(String::from),
                confidence: word.confidence,
            })
            .collect();
        Ok(transcription::TranscriptionResponse {
            usage,
            language: self.language,
            duration: self.duration,
            confidence: self.confidence,
            segments,
            words,
            ..transcription::TranscriptionResponse::new(self.text)
        })
    }
}

#[cfg(test)]
mod tests;
