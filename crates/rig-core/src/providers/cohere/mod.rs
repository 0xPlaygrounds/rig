//! Cohere configuration, endpoint wires, and model identifiers.
//!
//! ```no_run
//! use rig_core::providers::cohere;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let provider = cohere::Cohere::from_env()?;
//!
//! let command_a = provider.completion(cohere::COMMAND_A_03_2025);
//! let embeddings = provider.embedding(cohere::EMBED_V4, None);
//! # Ok(())
//! # }
//! ```
//!
//! Pair a wire with a transport in a [`crate::Model`] to send it.

pub mod completion;
pub mod embeddings;
pub mod streaming;
pub mod wire;

pub use wire::{Chat, CohereConfig, Embeddings, ImageEmbeddings};

use crate::client::macros::http_client;
use crate::driver::Model;

/// `command-a-plus-05-2026` completion model
pub const COMMAND_A_PLUS_05_2026: &str = "command-a-plus-05-2026";
/// `command-a-03-2025` completion model
pub const COMMAND_A_03_2025: &str = "command-a-03-2025";
/// `command-a-reasoning-08-2025` completion model
pub const COMMAND_A_REASONING_08_2025: &str = "command-a-reasoning-08-2025";
/// `command-a-vision-07-2025` completion model
pub const COMMAND_A_VISION_07_2025: &str = "command-a-vision-07-2025";
/// `command-a-translate-08-2025` completion model
pub const COMMAND_A_TRANSLATE_08_2025: &str = "command-a-translate-08-2025";
/// `command-r7b-12-2024` completion model
pub const COMMAND_R7B_12_2024: &str = "command-r7b-12-2024";
/// `command-r-plus-08-2024` completion model
pub const COMMAND_R_PLUS_08_2024: &str = "command-r-plus-08-2024";
/// `command-r-08-2024` completion model
pub const COMMAND_R_08_2024: &str = "command-r-08-2024";

/// `embed-v4.0` embedding model
pub const EMBED_V4: &str = "embed-v4.0";
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
        EMBED_V4 => Some(1_536),
        EMBED_ENGLISH_V3 | EMBED_MULTILINGUAL_V3 => Some(1_024),
        EMBED_ENGLISH_LIGHT_V3 | EMBED_MULTILINGUAL_LIGHT_V3 => Some(384),
        _ => None,
    }
}

http_client!(
    /// Cohere: its [`CohereConfig`] on a transport. Every model it builds
    /// sends through that transport.
    Cohere,
    CohereConfig
);

impl Cohere {
    /// Cohere with `api_key` and default settings, on the shared reqwest
    /// client.
    #[cfg(feature = "reqwest")]
    #[cfg_attr(docsrs, doc(cfg(feature = "reqwest")))]
    pub fn new(api_key: impl Into<crate::wire::Secret>) -> Self {
        CohereConfig::new(api_key).client()
    }

    /// Cohere from `COHERE_API_KEY`, on the shared reqwest client.
    #[cfg(feature = "reqwest")]
    #[cfg_attr(docsrs, doc(cfg(feature = "reqwest")))]
    pub fn from_env() -> Result<Self, crate::client::env::EnvError> {
        Ok(CohereConfig::from_env()?.client())
    }

    /// The chat model for `model`.
    pub fn completion(&self, model: impl Into<String>) -> Model<Chat> {
        self.model(self.config.completion(model))
    }

    /// The text-embedding model for `model`. `ndims` is the width it
    /// reports, defaulting to the model's known width.
    pub fn embedding(&self, model: impl Into<String>, ndims: Option<usize>) -> Model<Embeddings> {
        self.model(self.config.embedding(model, ndims))
    }

    /// The image-embedding model. Cohere embeds images with one fixed model,
    /// so it names none.
    pub fn image_embedding(&self) -> Model<ImageEmbeddings> {
        self.model(self.config.image_embedding())
    }
}

#[cfg(test)]
mod tests;
