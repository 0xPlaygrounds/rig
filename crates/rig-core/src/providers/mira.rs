//! Mira gateway configuration through [`crate::providers::openai::wire::MIRA`].
//! Model identifiers come from its model-listing endpoint rather than constants.
//!
//! ```no_run
//! use rig_core::providers::openai::wire::{MIRA, OpenAI};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let provider = OpenAI::from_env_with(&MIRA)?;
//! let models = provider.models();
//! # let _ = models;
//! # Ok(())
//! # }
//! ```
