//! OpenRouter's transcription model identifiers.
//!
//! OpenRouter's speech-to-text route takes a JSON body carrying the audio as
//! a base64 `input_audio` block rather than a multipart upload. The gateway's
//! own usage fields (audio seconds, cost) stay readable on
//! [`TranscriptionResponse::raw`](crate::transcription::TranscriptionResponse::raw).
//!
//! Timed segments and words come back with `response_format: "verbose_json"`;
//! words also need `"word"` in `timestamp_granularities`. Models that cannot
//! produce them reject `verbose_json` with a 400. Speaker labels need the
//! serving provider's own diarization option under `provider.options`, keyed
//! by its slug, and arrive as indices such as `"0"`.
//!
//! ```no_run
//! use rig_core::providers::openrouter;
//! use rig_core::transcription::TranscriptionRequestBuilder;
//! use serde_json::json;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let model = openrouter::from_env()?.transcription("microsoft/mai-transcribe-2");
//! let request = TranscriptionRequestBuilder::from_file("call.mp3")?
//!     .additional_params(json!({
//!         "response_format": "verbose_json",
//!         "timestamp_granularities": ["segment", "word"],
//!         "provider": {"options": {"azure": {"diarization": {"enabled": true}}}},
//!     }))
//!     .build();
//! for segment in model.call(request).await?.segments {
//!     println!("{:?} {:.1}s: {}", segment.speaker, segment.start, segment.text);
//! }
//! # Ok(())
//! # }
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
