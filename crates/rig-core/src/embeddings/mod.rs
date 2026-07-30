//! Provider-agnostic embedding abstractions.
//!
//! Embeddings are numerical representations of text or other inputs. Rig uses
//! each provider's `functions::embed` to generate vectors, [`Embed`] to select
//! which text from a Rust value should be embedded, and
//! [`batching::embed_documents`] to batch embedding requests for vector stores
//! or retrieval workflows.

pub mod batching;
pub mod embed;
pub mod embedding;
pub mod tool;

pub mod distance;
pub use batching::{default_concurrency, embed_documents, embed_documents_with_usage};
pub use embed::{Embed, EmbedError, TextEmbedder, to_texts};
pub use embedding::*;
pub use tool::ToolSchema;
