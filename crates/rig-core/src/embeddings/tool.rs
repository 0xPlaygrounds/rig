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

use crate::{Embed, tool::PortableToolEmbedding};
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

impl ToolSchema {
    /// Captures a tool's name, context, and embedding descriptions.
    /// Returns an error if context serialization fails.
    ///
    /// ```
    /// use rig_core::{embeddings::{ToolSchema, EmbedError}, tool::PortableToolEmbedding};
    ///
    /// fn schema(tool: &impl PortableToolEmbedding) -> Result<ToolSchema, EmbedError> {
    ///     ToolSchema::try_from(tool)
    /// }
    /// ```
    pub fn try_from<T>(tool: &T) -> Result<Self, EmbedError>
    where
        T: PortableToolEmbedding,
    {
        Ok(ToolSchema {
            name: T::NAME.to_string(),
            context: serde_json::to_value(tool.context()).map_err(EmbedError::new)?,
            embedding_docs: tool.embedding_docs(),
        })
    }
}

#[cfg(test)]
mod tests;
