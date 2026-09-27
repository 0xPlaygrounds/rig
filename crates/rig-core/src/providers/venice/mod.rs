//! Venice model identifiers, request parameters, and typed provider responses.
//!
//! [`from_env`] and [`new`] build a client on the [`VENICE`](crate::providers::openai::wire::VENICE) dialect.
//! [`VeniceParameters`] supplies request extensions; [`CompletionResponse`]
//! decodes provider-specific fields from the normalized response's `raw` value.
//!
//! ```no_run
//! use rig_core::providers::venice;
//! let model = venice::from_env()?.chat(venice::QWEN3_5_9B);
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

crate::providers::internal::client::openai_vendor!(
    crate::providers::openai::wire::VENICE,
    "Venice"
);
