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

use std::{cmp::max, ops::Range};

use futures::{StreamExt, TryStreamExt, stream};

use crate::driver::DynModel;
use crate::error::ProviderError;
use crate::operation::{Embedding as EmbeddingOp, ImageEmbedding as ImageEmbeddingOp};
use crate::wasm_compat::WasmCompatSend;
use crate::wire::Operation;
use crate::{
    completion::Usage,
    embeddings::{
        Embed, EmbedError, Embedding, EmbeddingResponse,
        embed::{EmbedInput, Embedder},
    },
};

/// Accumulates documents and embeds their texts and images in
/// provider-sized batches. Extraction occurs when documents are added;
/// requests start when built.
#[must_use = "an embeddings builder does nothing until built"]
pub struct EmbeddingsBuilder<T> {
    model: DynModel<EmbeddingOp>,
    image_model: Option<DynModel<ImageEmbeddingOp>>,
    documents: Vec<(T, Vec<EmbedInput>)>,
}

impl<T: Embed> EmbeddingsBuilder<T> {
    /// Create a new embedding builder that embeds texts with `model`.
    pub fn new(model: impl Into<DynModel<EmbeddingOp>>) -> Self {
        Self {
            model: model.into(),
            image_model: None,
            documents: vec![],
        }
    }

    /// Embed images with `model`. A text query finds an image only when
    /// this model embeds into the text model's space, as a CLIP vision and
    /// text pair does.
    pub fn image_model(mut self, model: impl Into<DynModel<ImageEmbeddingOp>>) -> Self {
        self.image_model = Some(model.into());
        self
    }

    /// Add a document to be embedded to the builder. `document` must implement the [Embed] trait.
    pub fn document(mut self, document: T) -> Result<Self, EmbedError> {
        let mut embedder = Embedder::default();
        document.embed(&mut embedder)?;

        self.documents.push((document, embedder.inputs));

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
    T: Embed + WasmCompatSend,
{
    /// Generate embeddings for all documents in the builder.
    ///
    /// Returns `(document, embeddings)` pairs. A document may produce one or many
    /// embeddings depending on how its [`Embed`] implementation uses [`Embedder`].
    ///
    /// Preserves document insertion order and each document's input order,
    /// regardless of batch completion order. Providers must return embeddings
    /// in input order within each batch; this is not checked.
    ///
    /// Propagates provider and transport errors. Returns an error identifying
    /// the document if it embeds nothing, if it embeds an image and the builder
    /// has no image model, or if a batch returns too few embeddings. Empty
    /// embedded collections embed nothing. Surplus embeddings are ignored.
    pub async fn build(self) -> Result<Vec<(T, Vec<Embedding>)>, ProviderError> {
        let (result, _usage) = self.build_with_usage().await?;
        Ok(result)
    }

    /// Generate embeddings for all documents in the builder and return accumulated token usage.
    ///
    /// Returns `(document, embeddings)` pairs and the total token usage across all
    /// batches. A document may produce one or many embeddings depending on how its
    /// [`Embed`] implementation uses [`Embedder`].
    ///
    /// Ordering is guaranteed at both levels, and the errors originate here;
    /// both are described on [`Self::build`].
    pub(crate) async fn build_with_usage(
        self,
    ) -> Result<(Vec<(T, Vec<Embedding>)>, Usage), ProviderError> {
        // Per-input slots preserve order even when a document spans batches
        // that finish out of order, or mixes texts and images.
        let mut docs: Vec<T> = Vec::with_capacity(self.documents.len());
        let mut spans: Vec<Range<usize>> = Vec::with_capacity(self.documents.len());
        let mut texts: Vec<(usize, String)> = Vec::new();
        let mut images: Vec<(usize, Vec<u8>)> = Vec::new();
        let mut total = 0;

        for (index, (doc, inputs)) in self.documents.into_iter().enumerate() {
            if inputs.is_empty() {
                return Err(ProviderError::Response(format!(
                    "document {index} produced nothing to embed, so it has no \
                     embeddings to return; an empty collection in an `#[embed]` \
                     field embeds nothing"
                )));
            }
            let start = total;
            for input in inputs {
                match input {
                    EmbedInput::Text(text) => texts.push((total, text)),
                    EmbedInput::Image(bytes) if self.image_model.is_some() => {
                        images.push((total, bytes))
                    }
                    EmbedInput::Image(_) => {
                        return Err(ProviderError::request(format!(
                            "document {index} embeds an image, but the builder has no \
                             image model; set one with `image_model`"
                        )));
                    }
                }
                total += 1;
            }
            spans.push(start..total);
            docs.push(doc);
        }

        let ((text_embeddings, text_usage), (image_embeddings, image_usage)) =
            futures::try_join!(batched(&self.model, texts), async {
                match &self.image_model {
                    Some(model) => batched(model, images).await,
                    None => Ok((Vec::new(), Usage::default())),
                }
            },)?;

        let mut slots: Vec<Option<Embedding>> = (0..total).map(|_| None).collect();
        for (slot, embedding) in text_embeddings.into_iter().chain(image_embeddings) {
            // Assigned slots remain in range; zip discards any surplus
            // provider embeddings.
            if let Some(place) = slots.get_mut(slot) {
                *place = Some(embedding);
            }
        }

        let mut slots = slots.into_iter();
        let mut result = Vec::with_capacity(docs.len());

        for (index, (doc, span)) in docs.into_iter().zip(spans).enumerate() {
            // Missing slots identify short provider responses without silently
            // dropping a document's inputs.
            let embeddings = slots
                .by_ref()
                .take(span.len())
                .collect::<Option<Vec<Embedding>>>()
                .ok_or_else(|| {
                    ProviderError::Response(format!(
                        "provider returned fewer embeddings than inputs sent: \
                         document {index} is missing at least one of its {} inputs \
                         (slots {}..{} of {total})",
                        span.len(),
                        span.start,
                        span.end
                    ))
                })?;

            result.push((doc, embeddings));
        }

        Ok((result, text_usage + image_usage))
    }
}

/// Embed `inputs` with `model` in batches of its capacity, run
/// concurrently, pairing each embedding with its input's slot.
async fn batched<Op, I>(
    model: &DynModel<Op>,
    inputs: Vec<(usize, I)>,
) -> Result<(Vec<(usize, Embedding)>, Usage), ProviderError>
where
    Op: Operation<Request = Vec<I>, Response = EmbeddingResponse>,
    I: WasmCompatSend + 'static,
{
    let max_documents = max(1, model.capabilities().max_documents);
    stream::iter(inputs)
        .chunks(max_documents)
        .map(|chunk| async {
            let (slots, batch): (Vec<usize>, Vec<I>) = chunk.into_iter().unzip();
            let response = model.call(batch).await?;
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
            (Vec::new(), Usage::default()),
            |(mut all, mut usage), (chunk, chunk_usage)| async move {
                all.extend(chunk);
                usage += chunk_usage;
                Ok((all, usage))
            },
        )
        .await
}

#[cfg(test)]
mod tests;
