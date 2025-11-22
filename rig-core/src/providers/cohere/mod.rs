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

use crate::models;

// ================================================================
// Cohere Completion Models
// ================================================================

models! {
    pub enum CompletionModels {
        /// `command-r-plus` completion model
        CommandRPlus => "command-r-plus",
        /// `command-r` completion model
        CommandR => "command-r",
        /// `command` completion model
        Command => "command",
        /// `command-nightly` completion model
        CommandNightly => "command-nightly",
        /// `command-light` completion model
        CommandLight => "command-light",
        /// `command-light-nightly` completion model
        CommandLightNightly => "command-light-nightly",
    }
}
pub use CompletionModels::*;

// ================================================================
// Cohere Embedding Models
// ================================================================
models! {
    pub enum EmbeddingModels {
        /// `embed-english-v3.0` embedding model
        EmbedEnglish3 => "embed-english-v3.0",
        /// `embed-english-light-v3.0` embedding model
        EmbedEnglishLight3 => "embed-english-light-v3.0",
        /// `embed-multilingual-v3.0` embedding model
        EmbedMulti3 => "embed-multilingual-v3.0",
        /// `embed-multilingual-light-v3.0` embedding model
        EmbedMultiLight3 =>"embed-multilingual-light-v3.0",
        /// `embed-english-v2.0` embedding model
        EmbedEnglish2 => "embed-english-v2.0",
        /// `embed-english-light-v2.0` embedding model
        EmbedEnglishLight2 => "embed-english-light-v2.0",
        /// `embed-multilingual-v2.0` embedding model
        EmbedMulti2 => "embed-multilingual-v2.0",
    }
}
pub use EmbeddingModels::*;

impl EmbeddingModels {
    pub fn default_dimensions(&self) -> usize {
        use EmbeddingModels::*;

        match self {
            EmbedEnglish3 | EmbedMulti3 | EmbedEnglishLight2 => 1024,
            EmbedEnglishLight3 | EmbedMultiLight3 => 384,
            EmbedEnglish2 => 4096,
            EmbedMulti2 => 768,
        }
    }
}
