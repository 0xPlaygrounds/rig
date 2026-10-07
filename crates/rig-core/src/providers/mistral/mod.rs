//! Mistral's model identifiers.
//!
//! [`from_env`] and [`new`] build a client on the [`MISTRAL`](crate::providers::openai::wire::MISTRAL) dialect.
//! Provider fields of a reply stay in the response's `raw` value.
//!
//! ```no_run
//! use rig_core::providers::mistral;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let mistral = mistral::from_env()?;
//! let small = mistral.chat(mistral::MISTRAL_SMALL);
//! let embed = mistral.embedding(mistral::embedding::MISTRAL_EMBED, None);
//! # let _ = (small, embed);
//! # Ok(())
//! # }
//! ```

pub mod completion;
pub mod embedding;
pub mod extension;
pub mod transcription;

pub use completion::*;
pub use embedding::*;
pub use transcription::*;

/// The provider key: the dialect name, a reply's `Origin::provider` and
/// the typed provider-options key.
pub const PROVIDER_NAME: &str = "mistral";

crate::client::macros::openai_vendor!(crate::providers::openai::wire::MISTRAL, "Mistral");
