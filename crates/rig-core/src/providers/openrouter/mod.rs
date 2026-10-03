//! OpenRouter model identifiers.
//!
//! [`from_env`] and [`new`] build a client on the [`OPENROUTER`](crate::providers::openai::wire::OPENROUTER) dialect.
//! Routing preferences are the request's `provider` object, sent through
//! `additional_params`.
//!
//! ```no_run
//! use rig_core::completion::CompletionRequest;
//! use rig_core::providers::openrouter;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let sonar = openrouter::from_env()?.chat(openrouter::PERPLEXITY_SONAR_PRO);
//!
//! let request = CompletionRequest::new("What is Rig?")
//!     .additional_params(serde_json::json!({"provider": {"sort": "price"}}));
//! # let _ = (sonar, request);
//! # Ok(())
//! # }
//! ```

#[cfg(feature = "audio")]
#[cfg_attr(docsrs, doc(cfg(feature = "audio")))]
pub mod audio_generation;
pub mod completion;
pub mod transcription;

#[cfg(feature = "audio")]
pub use audio_generation::{GPT_4O_MINI_TTS, KOKORO_82M, VOXTRAL_MINI_TTS};
pub use completion::*;
pub use transcription::{
    CHIRP_3, GPT_4O_MINI_TRANSCRIBE, GPT_4O_TRANSCRIBE, WHISPER_1, WHISPER_LARGE_V3,
    WHISPER_LARGE_V3_TURBO,
};

crate::client::macros::openai_vendor!(crate::providers::openai::wire::OPENROUTER, "OpenRouter");
