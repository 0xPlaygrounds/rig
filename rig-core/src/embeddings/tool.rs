use crate::{
    tool::ToolEmbeddingDyn,
    vector_store::{VectorStoreError, VectorStoreIndex, VectorStoreIndexDyn},
    Embeddable, OneOrMany,
};
use serde::{Deserialize, Serialize};

use super::embeddable::EmbeddableError;

/// Used by EmbeddingsBuilder to embed anything that implements ToolEmbedding.
#[derive(Clone, Deserialize, Serialize, Default, Eq, PartialEq)]
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

impl<I> VectorStoreIndexDyn for I
where
    I: VectorStoreIndex<EmbeddableTool>,
{
    fn top_n<'a>(
        &'a self,
        query: &'a str,
        n: usize,
    ) -> futures::future::BoxFuture<'a, crate::vector_store::TopNResults> {
        Box::pin(async move {
            self.top_n(query, n)
                .await?
                .into_iter()
                .map(|(distance, id, tool)| {
                    Ok((
                        distance,
                        id,
                        serde_json::to_value(tool).map_err(VectorStoreError::JsonError)?,
                    ))
                })
                .collect::<Result<Vec<_>, _>>()
        })
    }

    fn top_n_ids<'a>(
        &'a self,
        query: &'a str,
        n: usize,
    ) -> futures::future::BoxFuture<
        'a,
        Result<Vec<(f64, String)>, crate::vector_store::VectorStoreError>,
    > {
        Box::pin(self.top_n_ids(query, n))
    }
}
