//! Embeddable schema used for vector retrieval of [`ToolEmbedding`] values.

use crate::{Embed, tool::ToolEmbedding};
use serde::Serialize;

use super::embed::EmbedError;

/// Stored representation of a retrievable tool.
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
    /// Convert a typed retrievable tool into its stored schema.
    pub fn try_from<T>(tool: &T) -> Result<Self, EmbedError>
    where
        T: ToolEmbedding,
    {
        Ok(Self {
            name: tool.name(),
            context: serde_json::to_value(tool.context()).map_err(EmbedError::new)?,
            embedding_docs: tool.embedding_docs(),
        })
    }

    pub(crate) fn from_parts(
        name: String,
        context: serde_json::Result<serde_json::Value>,
        embedding_docs: Vec<String>,
    ) -> Result<Self, EmbedError> {
        Ok(Self {
            name,
            context: context.map_err(EmbedError::new)?,
            embedding_docs,
        })
    }
}
