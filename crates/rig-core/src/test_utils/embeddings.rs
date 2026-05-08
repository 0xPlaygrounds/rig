//! Embedding helpers for deterministic tests.

use crate::{
    client::Nothing,
    embeddings::{Embedding, EmbeddingError, EmbeddingModel},
};

/// A deterministic [`EmbeddingModel`] that returns a fixed vector for each input document.
#[derive(Clone, Debug, Default)]
pub struct MockEmbeddingModel;

impl EmbeddingModel for MockEmbeddingModel {
    const MAX_DOCUMENTS: usize = 5;

    type Client = Nothing;

    fn make(_: &Self::Client, _: impl Into<String>, _: Option<usize>) -> Self {
        Self
    }

    fn ndims(&self) -> usize {
        10
    }

    async fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String> + Send,
    ) -> Result<Vec<Embedding>, EmbeddingError> {
        Ok(documents
            .into_iter()
            .map(|document| Embedding {
                document,
                vec: vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            })
            .collect())
    }
}
