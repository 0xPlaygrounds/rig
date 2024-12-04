//! This module provides functionality for working with embeddings.
//! Embeddings are numerical representations of documents or other objects, typically used in
//! natural language processing (NLP) tasks such as text classification, information retrieval,
//! and document similarity.

pub mod builder;
pub mod embed;
pub mod embedding;
pub mod tool;

pub mod distance;
pub use builder::EmbeddingsBuilder;
pub use embed::{to_texts, Embed, EmbedError, TextEmbedder};
pub use embedding::{Embedding, EmbeddingError, EmbeddingModel};
pub use tool::ToolSchema;
