//! Mistral's model identifiers and its own view of a reply.
//!
//! [`from_env`] and [`new`] build a client on the [`MISTRAL`](crate::providers::openai::wire::MISTRAL) dialect.
//! [`CompletionResponse`] reads provider fields from a normalized chat response's
//! `raw` value; transcription metadata remains in its response's `raw` value.
//!
//! ```no_run
//! use rig_core::providers::mistral;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let mistral = mistral::from_env()?;
//! let small = mistral.chat(mistral::MISTRAL_SMALL);
//! let embed = mistral.embedding(mistral::embedding::MISTRAL_EMBED);
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

crate::client::macros::openai_vendor!(crate::providers::openai::wire::MISTRAL, "Mistral");
