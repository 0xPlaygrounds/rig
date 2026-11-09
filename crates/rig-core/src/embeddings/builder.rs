//! Batched embedding generation for documents implementing [`Embed`].
//!
//! ```no_run
//! use rig_core::DynModel;
//! use rig_core::embeddings::EmbeddingsBuilder;
//! use rig_core::operation::Embedding;
//!
//! # async fn example(model: DynModel<Embedding>) -> Result<(), Box<dyn std::error::Error>> {
//! let documents = EmbeddingsBuilder::new(model)
//!     .documents(["first document", "second document"])?
//!     .build().await?;
//! # let _ = documents;
//! # Ok(())
//! # }
//! ```

use std::cmp::max;

use futures::{StreamExt, stream};

use crate::driver::DynModel;
use crate::embeddings::{Embed, EmbedError, Embedding, EmbeddingResponse, embed::TextEmbedder};
use crate::error::ProviderError;
use crate::operation::Embedding as EmbeddingOp;

/// Accumulates documents and embeds their extracted texts in provider-sized batches.
/// Text extraction occurs when documents are added; requests start when built.
#[must_use = "an embeddings builder does nothing until built"]
pub struct EmbeddingsBuilder<T> {
    model: DynModel<EmbeddingOp>,
    documents: Vec<(T, Vec<String>)>,
}

impl<T: Embed> EmbeddingsBuilder<T> {
    /// Create a new embedding builder with the given embedding model
    pub fn new(model: impl Into<DynModel<EmbeddingOp>>) -> Self {
        Self {
            model: model.into(),
            documents: vec![],
        }
    }

    /// Add a document to be embedded to the builder. `document` must implement the [Embed] trait.
    pub fn document(mut self, document: T) -> Result<Self, EmbedError> {
        let mut embedder = TextEmbedder::default();
        document.embed(&mut embedder)?;

        self.documents.push((document, embedder.texts));

        Ok(self)
    }

    /// Add multiple documents to be embedded to the builder. `documents` must be iterable
    /// with items that implement the [Embed] trait.
    pub fn documents(self, documents: impl IntoIterator<Item = T>) -> Result<Self, EmbedError> {
        let builder = documents
            .into_iter()
            .try_fold(self, EmbeddingsBuilder::document)?;

        Ok(builder)
    }
}

impl<T> EmbeddingsBuilder<T>
where
    T: Embed + crate::wasm_compat::WasmCompatSend,
{
    /// Generate embeddings for all documents in the builder.
    ///
    /// Returns `(document, embeddings)` pairs. A document may produce one or many
    /// embeddings depending on how its [`Embed`] implementation uses [`TextEmbedder`].
    ///
    /// Preserves document insertion order and each document's text order,
    /// regardless of batch completion order. Providers must return embeddings
    /// in input order within each batch; this is not checked.
    ///
    /// Propagates provider and transport errors, including a reply whose
    /// embedding count differs from the texts sent. Returns an error
    /// identifying the document if it produces no text; empty embedded
    /// collections produce no text.
    pub async fn build(self) -> Result<Vec<(T, Vec<Embedding>)>, ProviderError> {
        use stream::TryStreamExt;

        let mut docs: Vec<(T, usize)> = Vec::with_capacity(self.documents.len());
        let mut texts: Vec<String> = Vec::new();
        for (doc, doc_texts) in self.documents {
            docs.push((doc, doc_texts.len()));
            texts.extend(doc_texts);
        }

        // Batches finish in any order; numbering them keeps a document that
        // spans batches in order. Every reply holds one embedding per text.
        let model = &self.model;
        let max_documents = max(1, model.capabilities().max_documents);
        let mut batches: Vec<(usize, Vec<Embedding>)> = stream::iter(texts)
            .chunks(max_documents)
            .enumerate()
            .map(|(at, batch)| async move {
                let response: EmbeddingResponse = model.call(batch).await?;
                Ok::<_, ProviderError>((at, response.embeddings))
            })
            .buffer_unordered(max(1, 1024 / max_documents))
            .try_collect()
            .await?;
        batches.sort_unstable_by_key(|(at, _)| *at);
        let mut embeddings = batches.into_iter().flat_map(|(_, batch)| batch);

        let mut result = Vec::with_capacity(docs.len());
        for (index, (doc, len)) in docs.into_iter().enumerate() {
            if len == 0 {
                return Err(crate::error::ProviderError::Response(format!(
                    "document {index} produced no text to embed, so it has no \
                     embeddings to return; an empty collection in an `#[embed]` \
                     field embeds nothing"
                )));
            }
            result.push((doc, embeddings.by_ref().take(len).collect()));
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests;
