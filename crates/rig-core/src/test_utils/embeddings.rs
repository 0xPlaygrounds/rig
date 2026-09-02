//! Embedding helpers for deterministic tests.

use crate::{
    Embed,
    embeddings::{
        Embedding, EmbeddingError, EmbeddingModel, EmbeddingResponse,
        embed::{EmbedError, TextEmbedder},
    },
    wasm_compat::WasmCompatSend,
};

/// A deterministic [`EmbeddingModel`] that returns a fixed vector for each input document.
#[derive(Clone, Debug, Default)]
pub struct MockEmbeddingModel;

impl EmbeddingModel for MockEmbeddingModel {
    fn max_documents(&self) -> usize {
        5
    }

    fn ndims(&self) -> usize {
        10
    }

    async fn embed_texts_response(
        &self,
        documents: impl IntoIterator<Item = String> + WasmCompatSend,
    ) -> Result<EmbeddingResponse, EmbeddingError> {
        Ok(EmbeddingResponse::new(
            documents
                .into_iter()
                .map(|document| Embedding {
                    document,
                    vec: vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                })
                .collect(),
            "mock",
        ))
    }
}

/// A test document that contributes one text fragment to an embedding request.
#[derive(Clone, Debug)]
pub struct MockTextDocument {
    /// Stable document identifier used by tests.
    pub id: String,
    /// Text to embed.
    pub text: String,
}

impl MockTextDocument {
    /// Create a single-text embedding fixture.
    pub fn new(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text: text.into(),
        }
    }
}

impl Embed for MockTextDocument {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(self.text.clone());
        Ok(())
    }
}

/// A test document that contributes multiple text fragments to an embedding request.
#[derive(Clone, Debug)]
pub struct MockMultiTextDocument {
    /// Stable document identifier used by tests.
    pub id: String,
    /// Text fragments to embed.
    pub texts: Vec<String>,
}

impl MockMultiTextDocument {
    /// Create a multi-text embedding fixture.
    pub fn new(id: impl Into<String>, texts: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            id: id.into(),
            texts: texts.into_iter().map(Into::into).collect(),
        }
    }
}

impl Embed for MockMultiTextDocument {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        for text in &self.texts {
            embedder.embed(text.clone());
        }
        Ok(())
    }
}
