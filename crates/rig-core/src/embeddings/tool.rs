//! Embeddable tool descriptions and serialized tool context.
//!
//! ```
//! use rig_core::embeddings::ToolSchema;
//!
//! let schema = ToolSchema {
//!     name: "search".into(),
//!     embedding_docs: vec!["Search documents".into()],
//!     ..Default::default()
//! };
//! assert_eq!(schema.embedding_docs.len(), 1);
//! ```

use crate::Embed;
use serde::Serialize;

use super::embed::EmbedError;

/// Tool name, serialized context, and text descriptions for embedding-based retrieval.
#[derive(Clone, Serialize, Default, Eq, PartialEq)]
pub struct ToolSchema {
    pub name: String,
    pub context: serde_json::Value,
    pub embedding_docs: Vec<String>,
}

impl Embed for ToolSchema {
    fn embed(&self, embedder: &mut super::embed::TextEmbedder) -> Result<(), EmbedError> {
        for doc in &self.embedding_docs {
            embedder.embed(doc.clone());
        }
        Ok(())
    }
}
