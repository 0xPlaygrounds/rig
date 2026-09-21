//! Mistral's model identifiers and its own view of a reply.
//!
//! Configure requests with [`crate::providers::openai::wire::MISTRAL`].
//! [`CompletionResponse`] reads provider fields from a normalized chat response's
//! `raw` value; transcription metadata remains in its response's `raw` value.
//!
//! ```no_run
//! use rig_core::providers::mistral;
//! use rig_core::providers::openai::wire::{MISTRAL, OpenAI};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let mistral = OpenAI::from_env_with(&MISTRAL)?;
//! let small = mistral.chat(mistral::MISTRAL_SMALL);
//! let embed = mistral.embeddings(mistral::embedding::MISTRAL_EMBED, None);
//! # let _ = (small, embed);
//! # Ok(())
//! # }
//! ```

pub mod completion;
pub mod embedding;
pub mod transcription;

pub use completion::*;
pub use embedding::*;
pub use transcription::*;
