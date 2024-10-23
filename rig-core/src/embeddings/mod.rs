//! This module provides functionality for working with embeddings.
//! Embeddings are numerical representations of documents or other objects, typically used in
//! natural language processing (NLP) tasks such as text classification, information retrieval,
//! and document similarity.

pub mod builder;
pub mod extract_embedding_fields;
pub mod embedding;
pub mod tool;

pub use builder::EmbeddingsBuilder;
pub use extract_embedding_fields::ExtractEmbeddingFields;
pub use embedding::{Embedding, EmbeddingError, EmbeddingModel};
pub use tool::EmbeddableTool;
