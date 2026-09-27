//! Mistral's embedding model identifiers and its batching cap.
//!
//! ```no_run
//! use rig_core::providers::mistral;
//! let model = mistral::from_env()?.embedding(mistral::MISTRAL_EMBED, None);
//! # Ok::<(), rig_core::RigError>(())
//! ```

pub const MISTRAL_EMBED: &str = "mistral-embed";
/// Codestral embedding model with configurable output dimensions.
pub const CODESTRAL_EMBED: &str = "codestral-embed";

/// Maximum input count per embedding request.
pub const MAX_DOCUMENTS: usize = 256;
