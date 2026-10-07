//! Together AI's model identifiers.
//!
//! [`from_env`] and [`new`] build a client on the [`TOGETHER`](crate::providers::openai::wire::TOGETHER) dialect,
//! using `TOGETHER_API_KEY`.
//!
//! ```no_run
//! use rig_core::providers::together;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let together = together::from_env()?;
//! let embedding = together.embedding(together::BGE_BASE_EN_V1_5, None);
//! let chat = together.chat(together::MIXTRAL_8X7B_INSTRUCT_V0_1);
//! # let _ = (embedding, chat);
//! # Ok(())
//! # }
//! ```

pub mod completion;
pub mod embedding;
pub mod extension;

pub use completion::*;
pub use embedding::*;

/// The provider key: the dialect name, a reply's `Origin::provider` and
/// the typed provider-options key.
pub const PROVIDER_NAME: &str = "together";

crate::client::macros::openai_vendor!(crate::providers::openai::wire::TOGETHER, "Together AI");
