//! Venice's model identifiers, its own request block, and its own views of a
//! reply.
//!
//! [Venice](https://docs.venice.ai/overview/about-venice) is a privacy-focused
//! inference provider whose chat-completions endpoint is a drop-in replacement
//! for OpenAI's, so it has no client and no models of its own:
//! [`openai::wire::VENICE`](crate::providers::openai::wire::VENICE) carries
//! the base URL, the `VENICE_API_KEY`/`VENICE_BASE_URL` variables, and the
//! native `/image/generate` path.
//!
//! What lives here is data:
//!
//! - the model identifiers for chat ([`completion`]), embeddings
//!   ([`embedding`]), images ([`image_generation`], feature `image`), speech
//!   (`audio_generation`, feature `audio`) and transcription
//!   ([`transcription`]);
//! - [`VeniceParameters`] — Venice's own request block (web search, thinking
//!   control, characters), which rides on
//!   [`additional_params`](crate::completion::CompletionRequest::additional_params);
//! - [`CompletionResponse`] and [`ImageGenerationResponse`], the typed reads
//!   of Venice's own reply documents, including the web-search citations and
//!   per-request [`Cost`] the normalized response does not name.
//!
//! Venice's video, image-editing, music, web-augmentation (`/augment/*`),
//! crypto-RPC, character, API-key and billing endpoints have no corresponding
//! rig operation and are deliberately not modeled here.
//!
//! # Example
//! A wire is the config plus a model; `.bind(transport)` (or `.bound()` from
//! `rig-reqwest`) turns it into the model.
//! ```no_run
//! use rig_core::providers::openai::wire::{OpenAI, VENICE};
//! use rig_core::providers::venice;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! // From `VENICE_API_KEY` (and optionally `VENICE_BASE_URL`).
//! let model = OpenAI::from_env_with(&VENICE)?.chat(venice::QWEN3_5_9B);
//! # let _ = model;
//! # Ok(())
//! # }
//! ```
//!
//! # Venice-specific request parameters
//! ```no_run
//! use rig_core::completion::{CompletionRequestBuilder, CompletionResponse};
//! use rig_core::providers::venice;
//!
//! let request = CompletionRequestBuilder::unbound("What shipped in Rust this month?")
//!     .additional_params(
//!         venice::VeniceParameters::new()
//!             .enable_web_search(venice::WebSearchMode::Auto)
//!             .enable_web_citations(true)
//!             .into_additional_params(),
//!     )
//!     .build();
//!
//! // The reply's `raw` is Venice's own document, so its blocks — including
//! // the citations web search returns — read back through
//! // `venice::CompletionResponse`.
//! fn citations(response: CompletionResponse) -> serde_json::Result<()> {
//!     let venice: venice::CompletionResponse = serde_json::from_value(response.raw)?;
//!     for citation in venice.web_search_citations() {
//!         println!("{} — {}", citation.title, citation.url);
//!     }
//!     Ok(())
//! }
//! ```

/// Venice's API root, and the default base URL of
/// [`openai::wire::VENICE`](crate::providers::openai::wire::VENICE).
pub const VENICE_API_BASE_URL: &str = "https://api.venice.ai/api/v1";

#[cfg(feature = "audio")]
pub mod audio_generation;
pub mod completion;
pub mod embedding;
#[cfg(feature = "image")]
pub mod image_generation;
pub mod transcription;

#[cfg(feature = "audio")]
pub use audio_generation::*;
pub use completion::*;
pub use embedding::*;
#[cfg(feature = "image")]
pub use image_generation::*;
pub use transcription::*;
