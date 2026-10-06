//! Cohere configuration, model identifiers, and its chat and embedding
//! wires.
//!
//! Chat goes through [`CohereChat`]: by default Cohere's OpenAI
//! Compatibility API on the shared Chat Completions wire, as the
//! [`COHERE`](crate::providers::openai::wire::COHERE) dialect. A
//! [`ChatRoute`] opts in to Cohere's native chat API ([`NativeChat`]), which
//! grounds the answer in the request's documents and keeps the citations
//! that point at each text block in its provider item. Text and image
//! embeddings use Cohere's own `/v1/embed`.
//!
//! ```no_run
//! use rig_core::providers::cohere;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let provider = cohere::Cohere::from_env()?;
//!
//! let mut command_a = provider.completion(cohere::COMMAND_A_03_2025);
//! command_a.wire = command_a.wire.with_route(cohere::ChatRoute::Auto);
//! let embeddings = provider.embedding(cohere::EMBED_V4, None);
//! # Ok(())
//! # }
//! ```

pub mod chat;
pub mod embeddings;
pub mod streaming;
pub mod wire;

pub use crate::client::cohere::Cohere;
pub use chat::NativeChat;
pub use embeddings::{
    EMBED_ENGLISH_LIGHT_V3, EMBED_ENGLISH_V3, EMBED_MULTILINGUAL_LIGHT_V3, EMBED_MULTILINGUAL_V3,
    EMBED_V4, Embeddings, ImageEmbeddings,
};
pub use wire::{ChatRoute, CohereChat, CohereConfig};

/// Stable descriptor name recorded on normalized responses and telemetry.
pub(crate) const PROVIDER_NAME: &str = "cohere";

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

#[cfg(test)]
mod history_tests;
