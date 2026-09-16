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
//! typed read of OpenRouter's own chat reply document. Transcripts have no
//! typed read of their own: the gateway's extra usage fields stay on the
//! normalized response's `raw` value.
//!
//! # Example
//! A wire is the config plus a model; `.bind(transport)` (or `.bound()` from
//! `rig-reqwest`) turns it into the model.
//! ```no_run
//! use rig_core::completion::CompletionRequestBuilder;
//! use rig_core::providers::openai::wire::{OPENROUTER, OpenAI};
//! use rig_core::providers::openrouter::{self, ProviderPreferences};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let sonar = OpenAI::from_env_with(&OPENROUTER)?.chat(openrouter::PERPLEXITY_SONAR_PRO);
//!
//! // Routing preferences ride on the request, as `{"provider": …}`.
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
