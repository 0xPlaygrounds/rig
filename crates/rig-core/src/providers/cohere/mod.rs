//! Cohere API client and Rig integration
//!
//! # Example
//! ```no_run
//! use rig_core::{client::CompletionClient, providers::cohere};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let client = cohere::Client::new("YOUR_API_KEY")?;
//!
//! let command_a = client.completion_model(cohere::COMMAND_A_03_2025);
//! # Ok(())
//! # }
//! ```

pub mod client;
pub mod completion;
pub mod embeddings;
pub mod streaming;

pub use client::{ApiErrorResponse, ApiResponse, Client};
pub use completion::CompletionModel;
pub use embeddings::{EmbeddingModel, ImageEmbeddingModel};

// ================================================================
// Cohere Completion Models
// ================================================================

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

/// `command-r-plus` completion model
#[deprecated(
    note = "Cohere removed `command-r-plus` on 2025-09-15; requests using it fail. \
    Use `COMMAND_R_PLUS_08_2024`, `COMMAND_A_03_2025`, or `COMMAND_A_PLUS_05_2026` instead."
)]
pub const COMMAND_R_PLUS: &str = "command-r-plus";
/// `command-r` completion model
#[deprecated(
    note = "Cohere removed `command-r` on 2025-09-15; requests using it fail. \
    Use `COMMAND_R_08_2024`, `COMMAND_A_03_2025`, or `COMMAND_A_PLUS_05_2026` instead."
)]
pub const COMMAND_R: &str = "command-r";
/// `command` completion model
#[deprecated(
    note = "Cohere removed `command` on 2025-09-15; requests using it fail. \
    Use `COMMAND_R_08_2024`, `COMMAND_A_03_2025`, or `COMMAND_A_PLUS_05_2026` instead."
)]
pub const COMMAND: &str = "command";
/// `command-nightly` completion model
#[deprecated(
    note = "`command-nightly` still resolves but is absent from Cohere's published model \
    catalogue, so it carries no compatibility or availability guarantee. \
    Use `COMMAND_A_03_2025` or `COMMAND_A_PLUS_05_2026` instead."
)]
pub const COMMAND_NIGHTLY: &str = "command-nightly";
/// `command-light` completion model
#[deprecated(
    note = "Cohere removed `command-light` on 2025-09-15; requests using it fail. \
    Use `COMMAND_R7B_12_2024` or `COMMAND_A_03_2025` instead."
)]
pub const COMMAND_LIGHT: &str = "command-light";
/// `command-light-nightly` completion model
#[deprecated(
    note = "Cohere no longer serves `command-light-nightly`; requests using it return 404. \
    Use `COMMAND_R7B_12_2024` or `COMMAND_A_03_2025` instead."
)]
pub const COMMAND_LIGHT_NIGHTLY: &str = "command-light-nightly";

// ================================================================
// Cohere Embedding Models
// ================================================================

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_dimensions_cover_every_live_embed_model() {
        assert_eq!(model_dimensions_from_identifier(EMBED_V4), Some(1_536));
        assert_eq!(
            model_dimensions_from_identifier(EMBED_ENGLISH_V3),
            Some(1_024)
        );
        assert_eq!(
            model_dimensions_from_identifier(EMBED_MULTILINGUAL_V3),
            Some(1_024)
        );
        assert_eq!(
            model_dimensions_from_identifier(EMBED_ENGLISH_LIGHT_V3),
            Some(384)
        );
        assert_eq!(
            model_dimensions_from_identifier(EMBED_MULTILINGUAL_LIGHT_V3),
            Some(384)
        );
        assert_eq!(model_dimensions_from_identifier("embed-unknown"), None);
    }
}
