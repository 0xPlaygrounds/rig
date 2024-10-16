use crate::{self as rig, tool::ToolEmbeddingDyn};
use rig::embeddings::embeddable::Embeddable;
use rig_derive::Embeddable;
use serde::Serialize;

use super::embeddable::EmbeddableError;

/// Used by EmbeddingsBuilder to embed anything that implements ToolEmbedding.
#[derive(Embeddable, Clone, Serialize, Default, Eq, PartialEq)]
pub struct EmbeddableTool {
    name: String,
    context: serde_json::Value,
    #[embed]
    embedding_docs: Vec<String>
}

impl EmbeddableTool {
    /// Convert item that implements ToolEmbedding to an EmbeddableTool.
    pub fn try_from(tool: &dyn ToolEmbeddingDyn) -> Result<Self, EmbeddableError> {
        Ok(EmbeddableTool {
            name: tool.name(),
            context: serde_json::to_value(tool.context().map_err(EmbeddableError::SerdeError)?)
                .map_err(EmbeddableError::SerdeError)?,
            embedding_docs: tool.embedding_docs(),
        })
    }
}