//! Cohere API client and Rig integration
//!
//! # Example
//! ```
//! use rig::providers::cohere;
//!
//! let client = cohere::Client::new("YOUR_API_KEY");
//!
//! let command_r = client.completion_model(cohere::COMMAND_R);
//! ```

pub mod client;
pub mod completion;
pub mod embeddings;
pub mod streaming;

pub use client::{ApiErrorResponse, ApiResponse, Client};
pub use completion::CompletionModel;
pub use embeddings::EmbeddingModel;

// ================================================================
// Cohere Completion Models
// ================================================================

/// `command-r-plus` completion model
pub const COMMAND_R_PLUS: &str = "command-r-plus";
/// `command-r` completion model
pub const COMMAND_R: &str = "command-r";
/// `command` completion model
pub const COMMAND: &str = "command";
/// `command-nightly` completion model
pub const COMMAND_NIGHTLY: &str = "command-nightly";
/// `command-light` completion model
pub const COMMAND_LIGHT: &str = "command-light";
/// `command-light-nightly` completion model
pub const COMMAND_LIGHT_NIGHTLY: &str = "command-light-nightly";

// ================================================================
// Cohere Embedding Models
// ================================================================

/// `embed-english-v3.0` embedding model
pub const EMBED_ENGLISH_V3: &str = "embed-english-v3.0";
/// `embed-english-light-v3.0` embedding model
pub const EMBED_ENGLISH_LIGHT_V3: &str = "embed-english-light-v3.0";
/// `embed-multilingual-v3.0` embedding model
pub const EMBED_MULTILINGUAL_V3: &str = "embed-multilingual-v3.0";
/// `embed-multilingual-light-v3.0` embedding model
pub const EMBED_MULTILINGUAL_LIGHT_V3: &str = "embed-multilingual-light-v3.0";

pub(crate) fn model_dimensions_from_identifier(identifier: &str) -> Option<usize> {
    match identifier {
        EMBED_ENGLISH_V3 | EMBED_MULTILINGUAL_V3 => Some(1_024),
        EMBED_ENGLISH_LIGHT_V3 | EMBED_MULTILINGUAL_LIGHT_V3 => Some(384),
        _ => None,
    }
}
