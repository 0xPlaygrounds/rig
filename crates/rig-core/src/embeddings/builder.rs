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
/// ```no_run
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
pub struct EmbeddingsBuilder<M, T>
where
    M: EmbeddingModel,
    T: Embed,
{
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
            .try_fold(self, |builder, doc| builder.document(doc))?;

        Ok(builder)
    }
}

impl<M, T> EmbeddingsBuilder<M, T>
where
    M: EmbeddingModel,
    T: Embed + Send,
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
    /// - pairs come back in the order the documents were added, and
    /// - each document's embeddings come back in the order its [`Embed`] impl
    ///   produced the texts.
    ///
    /// This holds regardless of how the texts were batched or which batch the
    /// provider answered first. Positional callers depend on it — for example
    /// [`InMemoryVectorStore::add_documents`](crate::vector_store::in_memory_store::InMemoryVectorStore::add_documents)
    /// derives its document ids from this sequence — and both levels have been
    /// silently violated before (rig#2344, rig#2345), so treat the guarantee as
    /// load-bearing rather than incidental.
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
    /// Ordering is guaranteed at both levels; see [`Self::build`].
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

        // Compute the embeddings.
        let (slots, usage) = stream::iter(texts.into_iter().enumerate())
            // Chunk them into batches. Each batch size is at most the embedding API limit per request.
            .chunks(M::MAX_DOCUMENTS)
            // Generate the embeddings for each batch with usage tracking.
            .map(|chunk| async {
                let (slots, batch): (Vec<usize>, Vec<String>) = chunk.into_iter().unzip();

                let response: EmbeddingResponse = self.model.embed_texts_with_usage(batch).await?;
                Ok::<_, EmbeddingError>((
                    slots
                        .into_iter()
                        .zip(response.embeddings)
                        .collect::<Vec<_>>(),
                    response.usage,
                ))
            })
            // Parallelize the embeddings generation over 10 concurrent requests
            .buffer_unordered(max(1, 1024 / M::MAX_DOCUMENTS))
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
                        // The slot came from this function's own `enumerate`,
                        // so it is always in range. `get_mut` keeps that an
                        // assumption the compiler checks rather than a panic
                        // waiting on a provider that answers with more
                        // embeddings than it was sent; a dropped write would
                        // leave the slot empty and surface below as a located
                        // error instead.
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

        for (doc, span) in docs.into_iter().zip(spans) {
            // A document that embedded no text has no embeddings to return;
            // this has always been an error rather than an empty list.
            if span.is_empty() {
                return Err(crate::embeddings::EmbeddingError::ResponseError(
                    "missing embedding for document after batch merge".to_string(),
                ));
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
                        "provider returned fewer embeddings than texts sent; \
                         document covering texts {}..{} is incomplete",
                        span.start, span.end
                    ))
                })?;

            result.push((doc, embeddings));
        }

        Ok((result, usage))
    }
}

#[cfg(test)]
mod tests {
    use crate::embeddings::embed::{EmbedError, TextEmbedder};
    use crate::embeddings::{Embed, Embedding, EmbeddingError, EmbeddingModel};
    use crate::test_utils::{MockEmbeddingModel, MockMultiTextDocument, MockTextDocument};

    use super::EmbeddingsBuilder;

    fn definitions_multiple_text() -> Vec<MockMultiTextDocument> {
        vec![
            MockMultiTextDocument::new(
                "doc0",
                [
                    "A green alien that lives on cold planets.",
                    "A fictional digital currency that originated in the animated series Rick and Morty.",
                ],
            ),
            MockMultiTextDocument::new(
                "doc1",
                [
                    "An ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.",
                    "A fictional creature found in the distant, swampy marshlands of the planet Glibbo in the Andromeda galaxy.",
                ],
            ),
        ]
    }

    fn definitions_multiple_text_2() -> Vec<MockMultiTextDocument> {
        vec![
            MockMultiTextDocument::new("doc2", ["Another fake definitions"]),
            MockMultiTextDocument::new("doc3", ["Some fake definition"]),
        ]
    }

    fn definitions_single_text() -> Vec<MockTextDocument> {
        vec![
            MockTextDocument::new("doc0", "A green alien that lives on cold planets."),
            MockTextDocument::new(
                "doc1",
                "An ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.",
            ),
        ]
    }

    #[tokio::test]
    async fn test_build_multiple_text() {
        let fake_definitions = definitions_multiple_text();

        let fake_model = MockEmbeddingModel;
        let result = EmbeddingsBuilder::new(fake_model)
            .documents(fake_definitions)
            .unwrap()
            .build()
            .await
            .unwrap();

        assert_eq!(result.len(), 2);

        let first_definition = &result[0];
        assert_eq!(first_definition.0.id, "doc0");
        assert_eq!(first_definition.1.len(), 2);
        assert_eq!(
            first_definition.1.first().map(|e| e.document.as_str()),
            Some("A green alien that lives on cold planets.")
        );

        let second_definition = &result[1];
        assert_eq!(second_definition.0.id, "doc1");
        assert_eq!(second_definition.1.len(), 2);
        assert_eq!(
            second_definition.1.get(1).map(|e| e.document.as_str()),
            Some(
                "A fictional creature found in the distant, swampy marshlands of the planet Glibbo in the Andromeda galaxy."
            )
        )
    }

    #[tokio::test]
    async fn test_build_single_text() {
        let fake_definitions = definitions_single_text();

        let fake_model = MockEmbeddingModel;
        let result = EmbeddingsBuilder::new(fake_model)
            .documents(fake_definitions)
            .unwrap()
            .build()
            .await
            .unwrap();

        assert_eq!(result.len(), 2);

        let first_definition = &result[0];
        assert_eq!(first_definition.0.id, "doc0");
        assert_eq!(first_definition.1.len(), 1);
        assert_eq!(
            first_definition.1.first().map(|e| e.document.as_str()),
            Some("A green alien that lives on cold planets.")
        );

        let second_definition = &result[1];
        assert_eq!(second_definition.0.id, "doc1");
        assert_eq!(second_definition.1.len(), 1);
        assert_eq!(
            second_definition.1.first().map(|e| e.document.as_str()),
            Some(
                "An ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land."
            )
        )
    }

    #[tokio::test]
    async fn test_build_multiple_and_single_text() {
        let fake_definitions = definitions_multiple_text();
        let fake_definitions_single = definitions_multiple_text_2();

        let fake_model = MockEmbeddingModel;
        let result = EmbeddingsBuilder::new(fake_model)
            .documents(fake_definitions)
            .unwrap()
            .documents(fake_definitions_single)
            .unwrap()
            .build()
            .await
            .unwrap();

        assert_eq!(result.len(), 4);

        let second_definition = &result[1];
        assert_eq!(second_definition.0.id, "doc1");
        assert_eq!(second_definition.1.len(), 2);
        assert_eq!(
            second_definition.1.first().map(|e| e.document.as_str()),
            Some(
                "An ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land."
            )
        );

        let third_definition = &result[2];
        assert_eq!(third_definition.0.id, "doc2");
        assert_eq!(third_definition.1.len(), 1);
        assert_eq!(
            third_definition.1.first().map(|e| e.document.as_str()),
            Some("Another fake definitions")
        )
    }

    #[tokio::test]
    async fn test_build_string() {
        let bindings = definitions_multiple_text();
        let fake_definitions = bindings.iter().map(|def| def.texts.clone());

        let fake_model = MockEmbeddingModel;
        let result = EmbeddingsBuilder::new(fake_model)
            .documents(fake_definitions)
            .unwrap()
            .build()
            .await
            .unwrap();

        assert_eq!(result.len(), 2);

        let first_definition = &result[0];
        assert_eq!(first_definition.1.len(), 2);
        assert_eq!(
            first_definition.1.first().map(|e| e.document.as_str()),
            Some("A green alien that lives on cold planets.")
        );

        let second_definition = &result[1];
        assert_eq!(second_definition.1.len(), 2);
        assert_eq!(
            second_definition.1.get(1).map(|e| e.document.as_str()),
            Some(
                "A fictional creature found in the distant, swampy marshlands of the planet Glibbo in the Andromeda galaxy."
            )
        )
    }

    #[tokio::test]
    async fn test_build_preserves_input_order_across_batches() {
        // More documents than MockEmbeddingModel::MAX_DOCUMENTS (5) to exercise
        // the chunked, buffered batch path, and assert that the returned
        // sequence matches the input order exactly.
        let texts: Vec<String> = (0..12).map(|i| format!("text-{i:02}")).collect();

        let fake_model = MockEmbeddingModel;
        let result = EmbeddingsBuilder::new(fake_model)
            .documents(texts.clone())
            .unwrap()
            .build()
            .await
            .unwrap();

        assert_eq!(result.len(), texts.len());
        for (i, (doc, embeddings)) in result.into_iter().enumerate() {
            assert_eq!(doc, texts[i]);
            assert_eq!(embeddings.len(), 1);
            assert_eq!(embeddings[0].document, texts[i]);
        }
    }

    /// A model whose *first* batch is slow, so later batches finish first.
    ///
    /// `buffer_unordered` yields batches as they complete, which is the only
    /// way to observe rig#2345 deterministically: without a delay the batches
    /// happen to finish in submission order and the defect hides.
    #[derive(Clone)]
    struct SlowFirstBatchModel {
        calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl SlowFirstBatchModel {
        fn new() -> Self {
            Self {
                calls: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            }
        }
    }

    impl EmbeddingModel for SlowFirstBatchModel {
        const MAX_DOCUMENTS: usize = 5;

        type Client = crate::client::Nothing;

        fn make(_: &Self::Client, _: impl Into<String>, _: Option<usize>) -> Self {
            Self::new()
        }

        fn ndims(&self) -> usize {
            10
        }

        async fn embed_texts(
            &self,
            documents: impl IntoIterator<Item = String> + crate::wasm_compat::WasmCompatSend,
        ) -> Result<Vec<Embedding>, EmbeddingError> {
            let nth = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if nth == 0 {
                tokio::time::sleep(std::time::Duration::from_millis(150)).await;
            }
            Ok(documents
                .into_iter()
                .map(|document| Embedding {
                    document,
                    vec: vec![0.0; 10],
                })
                .collect())
        }
    }

    /// A document contributing `n` texts named `t0..t{n-1}`.
    #[derive(Debug)]
    struct NTexts(usize);

    impl Embed for NTexts {
        fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
            for i in 0..self.0 {
                embedder.embed(format!("t{i}"));
            }
            Ok(())
        }
    }

    /// rig#2345 — a document whose texts straddle a `MAX_DOCUMENTS` boundary
    /// must get its embeddings back in text order.
    ///
    /// Six texts against a limit of 5 splits into `[t0..t4]` and `[t5]`; the
    /// delayed first batch makes the trailing one finish first. Before the slot
    /// index this returned `["t5", "t0", "t1", "t2", "t3", "t4"]`.
    #[tokio::test]
    async fn test_build_preserves_text_order_within_a_straddling_document() {
        let result = EmbeddingsBuilder::new(SlowFirstBatchModel::new())
            .document(NTexts(6))
            .unwrap()
            .build()
            .await
            .unwrap();

        assert_eq!(result.len(), 1);
        let order: Vec<&str> = result[0]
            .1
            .iter()
            .map(|embedding| embedding.document.as_str())
            .collect();
        assert_eq!(order, ["t0", "t1", "t2", "t3", "t4", "t5"]);
    }

    /// The same guarantee across several straddling documents at once: every
    /// document's texts land in its own list, in order, with none borrowed
    /// from a neighbour.
    #[tokio::test]
    async fn test_build_preserves_text_order_across_many_straddling_documents() {
        // 4 documents x 3 texts = 12 texts over a limit of 5, so documents 1
        // and 2 both straddle a boundary.
        let docs: Vec<NTexts> = (0..4).map(|_| NTexts(3)).collect();

        let result = EmbeddingsBuilder::new(SlowFirstBatchModel::new())
            .documents(docs)
            .unwrap()
            .build()
            .await
            .unwrap();

        assert_eq!(result.len(), 4);
        for (_, embeddings) in &result {
            let order: Vec<&str> = embeddings
                .iter()
                .map(|embedding| embedding.document.as_str())
                .collect();
            assert_eq!(order, ["t0", "t1", "t2"]);
        }
    }

    /// A document that embeds no text has no embeddings to return. This has
    /// always been an error rather than an empty list; the slot rewrite keeps
    /// it that way.
    #[tokio::test]
    async fn test_build_rejects_a_document_that_embeds_no_text() {
        let error = EmbeddingsBuilder::new(MockEmbeddingModel)
            .document(NTexts(0))
            .unwrap()
            .build()
            .await
            .expect_err("a document with no texts has no embeddings");

        assert!(
            error.to_string().contains("missing embedding for document"),
            "unexpected error: {error}"
        );
    }
}
