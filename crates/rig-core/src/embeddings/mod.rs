//! Provider-agnostic embedding abstractions.
//!
//! Embeddings are numerical representations of text or other inputs. Rig uses
//! [`EmbeddingModel`] to generate vectors, [`Embed`] to select which text from a
//! Rust value should be embedded, and [`EmbeddingsBuilder`] to batch embedding
//! requests for retrieval workflows.

pub mod builder;
pub mod embed;
pub mod embedding;

pub mod distance;
pub use builder::EmbeddingsBuilder;
pub use embed::{Embed, EmbedError, TextEmbedder, to_texts};
pub use embedding::*;
