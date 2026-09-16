//! OpenRouter's model identifiers, its routing preferences, and its own view
//! of a reply.
//!
//! OpenRouter is an OpenAI chat-completions dialect, so it has no client and
//! no models of its own:
//! [`openai::wire::OPENROUTER`](crate::providers::openai::wire::OPENROUTER)
//! carries the base URL, the `OPENROUTER_API_KEY` variable, the `/key`
//! credential check, and the gateway's own rewrites.
//!
//! What lives here is data: the model identifiers ([`completion`],
//! [`transcription`], and `audio_generation` under feature `audio`), the
//! [`ProviderPreferences`] request block, and [`CompletionResponse`] — the
//! typed read of OpenRouter's own reply document.
//!
//! # Example
//! ```ignore
//! use rig_core::prelude::*;
//! use rig_core::providers::openai::wire::{OPENROUTER, OpenAI};
//! use rig_core::providers::openrouter::{self, ProviderPreferences};
//! use rig_reqwest::DefaultTransport;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let sonar = OpenAI::from_env_with(&OPENROUTER)?
//!     .bound()?
//!     .completion(openrouter::PERPLEXITY_SONAR_PRO);
//!
//! // Routing preferences ride on the request, as `{"provider": …}`.
//! let request = sonar
//!     .completion_request("What is Rig?")
//!     .additional_params(ProviderPreferences::new().cheapest().to_json())
//!     .build();
//! # let _ = request;
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
    CHIRP_3, GPT_4O_MINI_TRANSCRIBE, GPT_4O_TRANSCRIBE, TranscriptionResponse, WHISPER_1,
    WHISPER_LARGE_V3, WHISPER_LARGE_V3_TURBO,
};
