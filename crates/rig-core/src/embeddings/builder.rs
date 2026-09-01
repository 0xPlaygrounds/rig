//! The module defines the [EmbeddingsBuilder] struct which accumulates objects to be embedded
//! and batch generates the embeddings for each object when built.
//! Only types that implement the [Embed] trait can be added to the [EmbeddingsBuilder].

use std::{cmp::max, ops::Range};

use futures::{StreamExt, stream};

use crate::{
    completion::Usage,
    embeddings::{
        Embed, EmbedError, Embedding, EmbeddingError, EmbeddingModel, EmbeddingResponse,
        embed::TextEmbedder,
    },
};

/// Builder for creating embeddings from one or more documents of type `T`.
/// Note: `T` can be any type that implements the [Embed] trait.
///
/// Using the builder is preferred over using [EmbeddingModel::embed_text] directly as
/// it will batch the documents in a single request to the model provider.
///
/// # Example
/// ```ignore
/// use rig_core::{
///     client::{EmbeddingsClient, ProviderClient},
///     embeddings::EmbeddingsBuilder,
///     providers::openai,
/// };
///
/// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
/// // Create OpenAI client
/// let openai_client = openai::Client::from_env()?;
///
/// let model = openai_client.embedding_model(openai::TEXT_EMBEDDING_3_SMALL);
///
/// let embeddings = EmbeddingsBuilder::new(model.clone())
///     .documents(vec![
///         "1. *flurbo* (noun): A green alien that lives on cold planets.".to_string(),
///         "2. *flurbo* (noun): A fictional digital currency.".to_string(),
///         "1. *glarb-glarb* (noun): An ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.".to_string(),
///         "2. *glarb-glarb* (noun): A fictional creature from marshlands.".to_string(),
///         "1. *linlingdong* (noun): A term used by inhabitants of the sombrero galaxy to describe humans.".to_string(),
///         "2. *linlingdong* (noun): A rare instrument.".to_string(),
///     ])?
///     .build()
///     .await?;
/// # Ok(())
/// # }
/// ```
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
    /// # Order
    ///
    /// Both levels are ordered, and callers may rely on it:
    ///
    /// - pairs come back in the order the documents were added — positional
    ///   callers depend on this, for example
    ///   [`InMemoryVectorStore::add_documents`](crate::vector_store::in_memory_store::InMemoryVectorStore::add_documents),
    ///   which derives its document ids from this sequence; and
    /// - each document's embeddings come back in the order its [`Embed`] impl
    ///   produced the texts.
    ///
    /// Neither depends on how the texts were batched or on which batch the
    /// provider answered first. Both have been silently violated before
    /// (rig#2344, rig#2345), so treat the guarantee as load-bearing rather than
    /// incidental.
    ///
    /// The second bullet inherits one assumption this type cannot check:
    /// providers pair a batch's embeddings to its texts positionally, so a
    /// provider that reordered *within* a single response would still be
    /// believed. That is the provider's contract, not this builder's.
    ///
    /// # Errors
    ///
    /// Alongside whatever the provider and the transport return, two cases
    /// originate here:
    ///
    /// - **A document that produces no text** fails the whole build rather than
    ///   coming back with an empty list. This is easy to hit by accident: an
    ///   empty collection in an `#[embed]` field embeds nothing, because
    ///   [`Embed`] is implemented for `Vec<T>` element-wise.
    /// - **A provider returning fewer embeddings than the texts it was sent**
    ///   fails rather than handing back a short list, since a short list cannot
    ///   be told apart from a document that legitimately has fewer texts.
    ///
    /// Both name the offending document.
    pub async fn build(self) -> Result<Vec<(T, Vec<Embedding>)>, EmbeddingError> {
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
    pub async fn build_with_usage(
        self,
    ) -> Result<(Vec<(T, Vec<Embedding>)>, Usage), EmbeddingError> {
        use stream::TryStreamExt;

        // Flatten every document's texts into one slot-indexed list, recording
        // the contiguous slot range each document owns.
        //
        // The slot index is what makes ordering independent of completion
        // order at *both* levels. Keying by document alone was not enough
        // (rig#2345): `chunks` splits on a flat text count, so one document's
        // texts can straddle a batch boundary, `buffer_unordered` yields
        // batches as they finish, and appending to a per-document list then
        // recorded completion order — a straddling document got its own
        // embeddings back shuffled. A batch now writes each embedding into its
        // own slot, so when a batch finishes cannot affect where anything
        // lands.
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

        // Compute the embeddings.
        let (slots, usage) = stream::iter(texts.into_iter().enumerate())
            // Chunk them into batches. Each batch size is at most the embedding API limit per request.
            .chunks(max_documents)
            // Generate the embeddings for each batch with usage tracking.
            .map(|chunk| async {
                let (slots, batch): (Vec<usize>, Vec<String>) = chunk.into_iter().unzip();

                let response: EmbeddingResponse = self.model.embed_texts_response(batch).await?;
                Ok::<_, EmbeddingError>((
                    slots
                        .into_iter()
                        .zip(response.embeddings)
                        .collect::<Vec<_>>(),
                    response.usage,
                ))
            })
            // Parallelize the embeddings generation over 10 concurrent requests
            .buffer_unordered(max(1, 1024 / max_documents))
            // Write each embedding into the slot its text came from, and
            // accumulate usage.
            .try_fold(
                (
                    (0..total_texts)
                        .map(|_| None)
                        .collect::<Vec<Option<Embedding>>>(),
                    Usage::default(),
                ),
                |(mut slots, mut usage_acc), (chunk_embeddings, chunk_usage)| async move {
                    for (slot, embedding) in chunk_embeddings {
                        // Every slot came from this function's own `enumerate`
                        // and the `zip` above truncates to the shorter side, so
                        // this index is in range by construction — including
                        // when a provider answers with more embeddings than it
                        // was sent. `get_mut` rather than `slots[slot]` only
                        // because `clippy::indexing_slicing` is denied here.
                        if let Some(place) = slots.get_mut(slot) {
                            *place = Some(embedding);
                        }
                    }
                    usage_acc += chunk_usage;
                    Ok((slots, usage_acc))
                },
            )
            .await?;

        // Hand each document the contiguous run of slots its texts occupied,
        // in text order.
        let mut slots = slots.into_iter();
        let mut result = Vec::with_capacity(docs.len());

        for (index, (doc, span)) in docs.into_iter().zip(spans).enumerate() {
            // A document that embedded no text has no embeddings to return;
            // this has always been an error rather than an empty list.
            if span.is_empty() {
                return Err(crate::embeddings::EmbeddingError::ResponseError(format!(
                    "document {index} produced no text to embed, so it has no \
                     embeddings to return; an empty collection in an `#[embed]` \
                     field embeds nothing"
                )));
            }

            // An empty slot means the provider returned fewer embeddings than
            // the texts sent in some batch. Previously `zip` dropped the
            // surplus texts and the document came back with a short list;
            // naming the slot turns silent loss into a located error.
            let embeddings = slots
                .by_ref()
                .take(span.len())
                .collect::<Option<Vec<Embedding>>>()
                .ok_or_else(|| {
                    crate::embeddings::EmbeddingError::ResponseError(format!(
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
