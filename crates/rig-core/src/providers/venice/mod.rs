//! Venice model identifiers, request parameters, and typed provider responses.
//!
//! Configure the provider with [`crate::providers::openai::wire::VENICE`].
//! [`VeniceParameters`] supplies request extensions; [`CompletionResponse`]
//! decodes provider-specific fields from the normalized response's `raw` value.
//!
//! ```no_run
//! use rig_core::providers::openai::wire::{OpenAI, VENICE};
//! use rig_core::providers::venice;
//! let wire = OpenAI::from_env_with(&VENICE)?.chat(venice::QWEN3_5_9B);
//! # Ok::<(), Box<dyn std::error::Error>>(())
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
