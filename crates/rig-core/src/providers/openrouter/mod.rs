//! OpenRouter model identifiers.
//!
//! [`from_env`] and [`new`] build a client on the [`OPENROUTER`](crate::providers::openai::wire::OPENROUTER) dialect.
//! Routing preferences, model fallbacks and the reply's cost are typed in
//! [`extension`].
//!
//! ```no_run
//! use rig_core::completion::{CompletionRequest, ProviderOptions};
//! use rig_core::providers::openrouter;
//! use rig_core::providers::openrouter::extension::{
//!     OpenRouterExt, OpenRouterOptions, ProviderPreferences, ProviderSortStrategy,
//! };
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let sonar = openrouter::from_env()?.chat(openrouter::PERPLEXITY_SONAR_PRO);
//!
//! let routing = OpenRouterOptions::new()
//!     .provider(ProviderPreferences::new().sort(ProviderSortStrategy::Price));
//! let request = CompletionRequest::new("What is Rig?")
//!     .provider_options(ProviderOptions::new().with::<OpenRouterExt>(&routing)?);
//! # let _ = (sonar, request);
//! # Ok(())
//! # }
//! ```

#[cfg(feature = "audio")]
#[cfg_attr(docsrs, doc(cfg(feature = "audio")))]
pub mod audio_generation;
pub mod completion;
pub mod extension;
pub mod transcription;

#[cfg(feature = "audio")]
pub use audio_generation::{GPT_4O_MINI_TTS, KOKORO_82M, VOXTRAL_MINI_TTS};
pub use completion::*;
pub use transcription::{
    CHIRP_3, GPT_4O_MINI_TRANSCRIBE, GPT_4O_TRANSCRIBE, WHISPER_1, WHISPER_LARGE_V3,
    WHISPER_LARGE_V3_TURBO,
};

/// The provider key: the dialect name, a reply's `Origin::provider` and
/// the typed provider-options key.
pub const PROVIDER_NAME: &str = "openrouter";

crate::client::macros::openai_vendor!(crate::providers::openai::wire::OPENROUTER, "OpenRouter");
