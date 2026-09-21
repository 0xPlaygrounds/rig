//! OpenRouter model identifiers, routing preferences, and typed chat responses.
//!
//! Configure requests with [`crate::providers::openai::wire::OPENROUTER`].
//! [`ProviderPreferences`] supplies the request's `provider` extension;
//! [`CompletionResponse`] reads provider-specific fields from `raw`.
//!
//! ```no_run
//! use rig_core::completion::CompletionRequestBuilder;
//! use rig_core::providers::openai::wire::{OPENROUTER, OpenAI};
//! use rig_core::providers::openrouter::{self, ProviderPreferences};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let sonar = OpenAI::from_env_with(&OPENROUTER)?.chat(openrouter::PERPLEXITY_SONAR_PRO);
//!
//! let request = CompletionRequestBuilder::unbound("What is Rig?")
//!     .additional_params(ProviderPreferences::new().cheapest().to_json())
//!     .build();
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
