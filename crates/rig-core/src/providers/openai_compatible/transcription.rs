//! Shared response types for OpenAI-compatible transcription endpoints.
//!
//! These types remain nameable when Azure, Groq, Hugging Face, or Venice is
//! enabled without the concrete OpenAI provider.

pub use crate::providers::internal::transcription::{
    DurationTag, TokensTag, TranscriptionInputTokenDetails, TranscriptionResponse,
    TranscriptionUsage,
};
