//! Cohere API integration.
//!
//! # Example
//! ```no_run
//! use rig_core::providers::cohere;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! # let request = rig_core::completion::CompletionRequest::from_prompt("hello");
//! let cfg = cohere::functions::Config::from_env(cohere::COMMAND_R)?;
//! let rt = rig_core::http_runtime::HttpRuntime::new();
//! let response = cohere::functions::complete(&cfg, &rt, request).await?;
//! # Ok(())
//! # }
//! ```

pub mod completion;
pub mod embeddings;
pub mod functions;
pub mod streaming;

crate::providers::client::define_http_client! {
    config = functions::Config,
    default_base_url = functions::DEFAULT_BASE_URL,
    api_key_required = true,
}
crate::providers::client::impl_http_embedding_config_factory!(Client, functions::EmbeddingConfig);

pub use embeddings::{ApiErrorResponse, ApiResponse};

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

/// Embedding width for a known Cohere embedding model.
///
/// Retained from the deleted classic `EmbeddingModel::ndims()` so callers
/// sizing a vector-store index can still resolve a model's dimensions.
pub fn model_dimensions_from_identifier(identifier: &str) -> Option<usize> {
    match identifier {
        EMBED_ENGLISH_V3 | EMBED_MULTILINGUAL_V3 => Some(1_024),
        EMBED_ENGLISH_LIGHT_V3 | EMBED_MULTILINGUAL_LIGHT_V3 => Some(384),
        _ => None,
    }
}
