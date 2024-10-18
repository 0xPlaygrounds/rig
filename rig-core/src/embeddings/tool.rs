use crate::{tool::ToolEmbeddingDyn, Embeddable, OneOrMany};
use serde::{Deserialize};

use super::embeddable::EmbeddableError;

/// Used by EmbeddingsBuilder to embed anything that implements ToolEmbedding.
#[derive(Clone, Deserialize, Default, Eq, PartialEq)]
pub struct EmbeddableTool {
    pub name: String,
    pub context: serde_json::Value,
    pub embedding_docs: Vec<String>,
}

impl Embeddable for EmbeddableTool {
    type Error = EmbeddableError;

    fn embeddable(&self) -> Result<OneOrMany<String>, Self::Error> {
        OneOrMany::many(self.embedding_docs.clone()).map_err(EmbeddableError::new)
    }
}

impl EmbeddableTool {
    /// Convert item that implements ToolEmbedding to an EmbeddableTool.
    pub fn try_from(tool: &dyn ToolEmbeddingDyn) -> Result<Self, EmbeddableError> {
        Ok(EmbeddableTool {
            name: tool.name(),
            context: tool.context().map_err(EmbeddableError::new)?,
            embedding_docs: tool.embedding_docs(),
        })
    }
}
