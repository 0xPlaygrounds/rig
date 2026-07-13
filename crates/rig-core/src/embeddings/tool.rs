//! Embeddable representation of a [`ToolEmbedding`].

use crate::{
    Embed,
    tool::{ErasedEmbeddingTool, ToolEmbedding},
};
use serde::Serialize;

use super::embed::EmbedError;

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
    /// Convert a typed embedding tool to its stored schema.
    pub fn try_from<T>(tool: &T) -> Result<Self, EmbedError>
    where
        T: ToolEmbedding + 'static,
    {
        Self::from_erased(crate::tool::Tool::name(tool), tool)
    }

    pub(crate) fn from_erased(
        name: impl Into<String>,
        tool: &dyn ErasedEmbeddingTool,
    ) -> Result<Self, EmbedError> {
        Ok(Self {
            name: name.into(),
            context: tool.context().map_err(EmbedError::new)?,
            embedding_docs: tool.embedding_docs(),
        })
    }
}
