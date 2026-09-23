//! Common model-construction, completion, embedding, tool, and vector-store imports.
//!
//! ```
//! use rig_core::prelude::*;
//!
//! let message = Message::user("Hello");
//! ```

pub use crate::completion::{CompletionModel, Message};
pub use crate::driver::{Bind, Bound, CompletionProvider};
pub use crate::error::ProviderError;

// The root re-export includes the derive macro when enabled.
pub use crate::Embed;
pub use crate::embeddings::{EmbeddingModel, EmbeddingsBuilder};

pub use crate::tool::PortableTool;

pub use crate::vector_store::VectorStoreIndex;
pub use crate::vector_store::in_memory_store::InMemoryVectorStore;
pub use crate::vector_store::request::VectorSearchRequest;
