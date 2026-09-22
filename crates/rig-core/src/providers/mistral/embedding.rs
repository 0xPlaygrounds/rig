//! Mistral's embedding model identifiers and its batching cap.
//!
//! ```no_run
//! use rig_core::providers::{mistral, openai::wire::{MISTRAL, OpenAI}};
//! let wire = OpenAI::from_env_with(&MISTRAL)?.embeddings(mistral::MISTRAL_EMBED, None);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub const MISTRAL_EMBED: &str = "mistral-embed";
/// Codestral embedding model with configurable output dimensions.
pub const CODESTRAL_EMBED: &str = "codestral-embed";

/// Maximum input count per embedding request.
pub const MAX_DOCUMENTS: usize = 256;
