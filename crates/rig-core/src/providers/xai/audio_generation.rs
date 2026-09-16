//! xAI's text-to-speech model identifier.
//!
//! The request runs on the shared OpenAI speech wire, whose
//! [`XAI`](crate::providers::openai::wire::XAI) dialect carries the `/v1/tts`
//! path and xAI's body — `voice_id` rather than `voice`, no model field, and
//! `eve` as the default voice.

// ================================================================
// xAI TTS API
// ================================================================
pub const TTS_1: &str = "tts-1";
