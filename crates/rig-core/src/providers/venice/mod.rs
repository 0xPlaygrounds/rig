//! Venice model identifiers and request parameters.
//!
//! [`from_env`] and [`new`] build a client on the [`VENICE`](crate::providers::openai::wire::VENICE) dialect.
//! Its `venice_parameters` request block goes through `additional_params`;
//! provider fields of a reply stay in the response's `raw` value.
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

crate::client::macros::openai_vendor!(crate::providers::openai::wire::VENICE, "Venice");
