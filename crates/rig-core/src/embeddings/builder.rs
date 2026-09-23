//! Batched embedding generation for documents implementing [`Embed`].
//!
//! ```no_run
//! use rig_core::embeddings::{EmbeddingModel, EmbeddingsBuilder};
//!
//! # async fn example(model: impl EmbeddingModel) -> Result<(), Box<dyn std::error::Error>> {
//! let documents = EmbeddingsBuilder::new(model)
//!     .documents(["first document", "second document"])?
//!     .build().await?;
//! # let _ = documents;
//! # Ok(())
//! # }
//! ```

use std::{cmp::max, ops::Range};

use futures::{StreamExt, stream};

use crate::error::ProviderError;
use crate::{
    completion::Usage,
    embeddings::{
        Embed, EmbedError, Embedding, EmbeddingModel, EmbeddingResponse, embed::TextEmbedder,
    },
};

/// Accumulates documents and embeds their extracted texts in provider-sized batches.
/// Text extraction occurs when documents are added; requests start when built.
#[must_use = "an embeddings builder does nothing until built"]
pub struct EmbeddingsBuilder<M, T> {
    model: M,
    documents: Vec<(T, Vec<String>)>,
}

impl<M, T> EmbeddingsBuilder<M, T>
where
    M: EmbeddingModel,
    T: Embed,
{
    /// Create a new embedding builder with the given embedding model
    pub fn new(model: M) -> Self {
        Self {
            model,
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

impl<M, T> EmbeddingsBuilder<M, T>
where
    M: EmbeddingModel,
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
    /// Propagates provider and transport errors. Returns an error identifying
    /// the document if it produces no text or a batch returns too few embeddings.
    /// Empty embedded collections produce no text. Surplus embeddings are ignored.
    pub async fn build(self) -> Result<Vec<(T, Vec<Embedding>)>, ProviderError> {
        let (result, _usage) = self.build_with_usage().await?;
        Ok(result)
    }

    /// Generate embeddings for all documents in the builder and return accumulated token usage.
    ///
    /// Returns `(document, embeddings)` pairs and the total token usage across all
    /// batches. A document may produce one or many embeddings depending on how its
    /// [`Embed`] implementation uses [`TextEmbedder`].
    ///
    /// Ordering is guaranteed at both levels, and the same two errors originate
    /// here; both are described on [`Self::build`].
    pub(crate) async fn build_with_usage(
        self,
    ) -> Result<(Vec<(T, Vec<Embedding>)>, Usage), ProviderError> {
        use stream::TryStreamExt;

        // Per-text slots preserve order even when a document spans batches
        // that finish out of order.
        let mut docs: Vec<T> = Vec::with_capacity(self.documents.len());
        let mut spans: Vec<Range<usize>> = Vec::with_capacity(self.documents.len());
        let mut texts: Vec<String> = Vec::new();

        for (doc, doc_texts) in self.documents {
            let start = texts.len();
            texts.extend(doc_texts);
            spans.push(start..texts.len());
            docs.push(doc);
        }

        let total_texts = texts.len();
        let max_documents = max(1, self.model.max_documents());

        let (slots, usage) = stream::iter(texts.into_iter().enumerate())
            .chunks(max_documents)
            .map(|chunk| async {
                let (slots, batch): (Vec<usize>, Vec<String>) = chunk.into_iter().unzip();

                let response: EmbeddingResponse = self.model.embed_texts_response(batch).await?;
                Ok::<_, ProviderError>((
                    slots
                        .into_iter()
                        .zip(response.embeddings)
                        .collect::<Vec<_>>(),
                    response.usage,
                ))
            })
            .buffer_unordered(max(1, 1024 / max_documents))
            .try_fold(
                (
                    (0..total_texts)
                        .map(|_| None)
                        .collect::<Vec<Option<Embedding>>>(),
                    Usage::default(),
                ),
                |(mut slots, mut usage_acc), (chunk_embeddings, chunk_usage)| async move {
                    for (slot, embedding) in chunk_embeddings {
                        // Enumerated slots remain in range; zip discards any
                        // surplus provider embeddings.
                        if let Some(place) = slots.get_mut(slot) {
                            *place = Some(embedding);
                        }
                    }
                    usage_acc += chunk_usage;
                    Ok((slots, usage_acc))
                },
            )
            .await?;

        let mut slots = slots.into_iter();
        let mut result = Vec::with_capacity(docs.len());

        for (index, (doc, span)) in docs.into_iter().zip(spans).enumerate() {
            if span.is_empty() {
                return Err(crate::error::ProviderError::Response(format!(
                    "document {index} produced no text to embed, so it has no \
                     embeddings to return; an empty collection in an `#[embed]` \
                     field embeds nothing"
                )));
            }

            // Missing slots identify short provider responses without silently
            // dropping a document's texts.
            let embeddings = slots
                .by_ref()
                .take(span.len())
                .collect::<Option<Vec<Embedding>>>()
                .ok_or_else(|| {
                    crate::error::ProviderError::Response(format!(
                        "provider returned fewer embeddings than texts sent: \
                         document {index} is missing at least one of its {} texts \
                         (slots {}..{} of {total_texts})",
                        span.len(),
                        span.start,
                        span.end
                    ))
                })?;

            result.push((doc, embeddings));
        }

        Ok((result, usage))
    }
}

#[cfg(test)]
mod tests;
