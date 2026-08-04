//! Chunking and batch-alignment helpers for the provider `functions`
//! embedding faces.
//!
//! Every provider's free `embed` function honors its descriptor's
//! `max_embedding_documents` by splitting the input into wire-sized chunks
//! and re-assembling one order-aligned [`EmbeddingResponse`]; every
//! `embed_batches` flattens caller batches, embeds once, and regroups the
//! results per batch. Both loops live here so the per-provider modules stay
//! pure request/parse code.
//!
//! On top of those, [`embed_documents`] is the document-level entry point
//! that replaced the old `EmbeddingsBuilder`: hand it a `Vec` of [`Embed`]
//! values and a closure that embeds one chunk of texts, and it flattens,
//! chunks, parallelizes, and re-associates the embeddings for you.
//!
//! [`EmbeddingJob`] is the fluent face of the same thing: accumulate
//! documents, take the batch size from a provider's descriptor with
//! [`EmbeddingJob::for_provider`], and supply the provider closure only at
//! [`EmbeddingJob::run`]. The job stores no model, config, or callback, so it
//! is plain data right up to the terminal.

use std::collections::BTreeMap;
use std::future::Future;

use futures::{StreamExt, TryStreamExt, stream};

use crate::OneOrMany;
use crate::completion::Usage;
use crate::embeddings::embed::TextEmbedder;
use crate::embeddings::{Embed, Embedding, EmbeddingError, EmbeddingResponse};
use crate::http_runtime::HttpRuntime;

/// Map a transport-level [`crate::completion::CompletionError`] from
/// [`HttpRuntime::send`] into the embedding error vocabulary.
fn transport_error(error: crate::completion::CompletionError) -> EmbeddingError {
    match error {
        crate::completion::CompletionError::HttpError(e) => EmbeddingError::HttpError(e),
        other => EmbeddingError::ProviderError(other.to_string()),
    }
}

/// Send `request` over `rt`, returning `(status, body)` with transport
/// failures mapped into [`EmbeddingError`].
pub(crate) async fn send(
    rt: &HttpRuntime,
    request: http::Request<Vec<u8>>,
) -> Result<(http::StatusCode, String), EmbeddingError> {
    rt.send(request).await.map_err(transport_error)
}

/// Drive `texts` through `build`/`parse` in chunks of at most
/// `max_documents`, concatenating embeddings in input order and summing
/// usage.
///
/// `build` is the provider's pure request builder for one chunk; `parse` is
/// its pure response parser (handed the chunk's documents so embeddings can
/// be zipped back onto their inputs).
pub(crate) async fn embed_chunked<B, P>(
    rt: &HttpRuntime,
    texts: Vec<String>,
    max_documents: Option<usize>,
    build: B,
    parse: P,
) -> Result<EmbeddingResponse, EmbeddingError>
where
    B: Fn(&[String]) -> Result<http::Request<Vec<u8>>, EmbeddingError>,
    P: Fn(http::StatusCode, &str, Vec<String>) -> Result<EmbeddingResponse, EmbeddingError>,
{
    let chunk_size = max_documents.unwrap_or(usize::MAX).max(1);
    let mut embeddings = Vec::with_capacity(texts.len());
    let mut usage = Usage::new();
    for chunk in texts.chunks(chunk_size) {
        let request = build(chunk)?;
        let (status, body) = send(rt, request).await?;
        let response = parse(status, &body, chunk.to_vec())?;
        usage += response.usage;
        embeddings.extend(response.embeddings);
    }
    Ok(EmbeddingResponse { embeddings, usage })
}

/// Regroup a flat, order-aligned embedding list back into per-batch
/// [`OneOrMany`] groups of the given `counts`.
///
/// Errors if any batch is empty or the embedding count does not match the
/// flattened input count.
#[doc(hidden)]
pub fn group_batches(
    counts: &[usize],
    embeddings: Vec<Embedding>,
) -> Result<Vec<OneOrMany<Embedding>>, EmbeddingError> {
    if embeddings.len() != counts.iter().sum::<usize>() {
        return Err(EmbeddingError::ResponseError(format!(
            "provider returned {} embeddings for {} input documents",
            embeddings.len(),
            counts.iter().sum::<usize>()
        )));
    }
    let mut iter = embeddings.into_iter();
    counts
        .iter()
        .map(|&count| {
            let group: Vec<Embedding> = iter.by_ref().take(count).collect();
            OneOrMany::many(group).map_err(|_| {
                EmbeddingError::DocumentError("cannot embed an empty batch of documents".into())
            })
        })
        .collect()
}

/// Split `batches` into per-batch counts plus the flattened text list, the
/// front half of the flatten → embed → [`group_batches`] pipeline.
pub(crate) fn split_batches(batches: Vec<Vec<String>>) -> (Vec<usize>, Vec<String>) {
    let counts: Vec<usize> = batches.iter().map(Vec::len).collect();
    let texts: Vec<String> = batches.into_iter().flatten().collect();
    (counts, texts)
}

/// The concurrency the retired `EmbeddingsBuilder` used: enough in-flight
/// requests to keep roughly 1024 documents on the wire at once, never fewer
/// than one.
///
/// Callers that do not care to tune concurrency can pass this straight into
/// [`embed_documents`].
pub fn default_concurrency(max_documents: usize) -> usize {
    (1024 / max_documents.max(1)).max(1)
}

/// Embed a list of [`Embed`] documents, returning each document paired with
/// its embeddings in input order.
///
/// Each document contributes zero or more texts via its [`Embed`]
/// implementation (see [`TextEmbedder`]). Those texts are flattened across
/// all documents, split into chunks of at most `max_documents`, embedded
/// with at most `concurrency` chunks in flight at once, and then regrouped
/// onto their originating documents.
///
/// * `max_documents` is the provider's per-request document limit — source it
///   from [`ProviderDescriptor::max_embedding_documents`](crate::providers::descriptor::ProviderDescriptor),
///   falling back to [`usize::MAX`] when the descriptor leaves it unset. A
///   value of `0` is treated as `1`.
/// * `concurrency` bounds the number of simultaneous `embed_batch` calls;
///   [`default_concurrency`] reproduces the historical builder behavior. A
///   value of `0` is treated as `1`.
/// * `embed_batch` embeds one chunk of texts and must return embeddings in
///   input order, one per input text. In practice this is a closure over a
///   provider's free `embed` function.
///
/// # Errors
/// * [`EmbeddingError::DocumentError`] if a document's [`Embed`]
///   implementation fails, or if a document yields **zero** texts (every
///   document must contribute at least one text).
/// * [`EmbeddingError::ResponseError`] if `embed_batch` returns a number of
///   embeddings that does not match the chunk it was handed.
/// * The first `embed_batch` error, short-circuiting the remaining chunks.
///
/// Ordering is guaranteed on both axes: the returned `Vec` is in `documents`
/// order, and each document's [`OneOrMany`] is in the order its [`Embed`]
/// implementation produced the texts — regardless of the order concurrent
/// chunks complete in.
///
/// # Example
/// ```no_run
/// use rig_core::embeddings::batching::{default_concurrency, embed_documents};
/// use rig_core::http_runtime::HttpRuntime;
/// use rig_core::providers::openai;
///
/// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
/// let cfg = openai::functions::EmbeddingConfig::from_env("text-embedding-3-small")?;
/// let rt = HttpRuntime::new();
///
/// let max_documents = openai::functions::DESCRIPTOR
///     .max_embedding_documents
///     .unwrap_or(usize::MAX);
///
/// let embeddings = embed_documents(
///     vec![
///         "1. *flurbo* (noun): A green alien that lives on cold planets.".to_string(),
///         "2. *flurbo* (noun): A fictional digital currency.".to_string(),
///     ],
///     max_documents,
///     default_concurrency(max_documents),
///     |texts| openai::functions::embed(&cfg, &rt, texts),
/// )
/// .await?;
///
/// for (document, embedding) in embeddings {
///     println!("{document}: {} vector(s)", embedding.len());
/// }
/// # Ok(())
/// # }
/// ```
pub async fn embed_documents<D, F, Fut>(
    documents: Vec<D>,
    max_documents: usize,
    concurrency: usize,
    embed_batch: F,
) -> Result<Vec<(D, OneOrMany<Embedding>)>, EmbeddingError>
where
    D: Embed,
    F: Fn(Vec<String>) -> Fut,
    Fut: Future<Output = Result<EmbeddingResponse, EmbeddingError>>,
{
    let (documents, _usage) =
        embed_documents_with_usage(documents, max_documents, concurrency, embed_batch).await?;
    Ok(documents)
}

/// [`embed_documents`], additionally returning the token usage summed across
/// every chunk.
pub async fn embed_documents_with_usage<D, F, Fut>(
    documents: Vec<D>,
    max_documents: usize,
    concurrency: usize,
    embed_batch: F,
) -> Result<(Vec<(D, OneOrMany<Embedding>)>, Usage), EmbeddingError>
where
    D: Embed,
    F: Fn(Vec<String>) -> Fut,
    Fut: Future<Output = Result<EmbeddingResponse, EmbeddingError>>,
{
    // Extract each document's texts up front so a malformed `Embed`
    // implementation fails before any request is sent.
    let mut counts = Vec::with_capacity(documents.len());
    let mut texts: Vec<String> = Vec::new();
    for document in &documents {
        let mut embedder = TextEmbedder::default();
        document
            .embed(&mut embedder)
            .map_err(|e| EmbeddingError::DocumentError(Box::new(e)))?;
        counts.push(embedder.texts.len());
        texts.extend(embedder.texts);
    }

    if texts.is_empty() {
        // Nothing to send. `group_batches` still rejects zero-text documents.
        let groups = group_batches(&counts, Vec::new())?;
        return Ok((documents.into_iter().zip(groups).collect(), Usage::new()));
    }

    let chunk_size = max_documents.max(1);
    let chunks: Vec<(usize, Vec<String>)> = texts
        .chunks(chunk_size)
        .map(<[String]>::to_vec)
        .enumerate()
        .collect();

    let (chunked, usage) = stream::iter(chunks)
        .map(|(index, chunk)| {
            let embed_batch = &embed_batch;
            async move {
                let expected = chunk.len();
                let response = embed_batch(chunk).await?;
                if response.embeddings.len() != expected {
                    return Err(EmbeddingError::ResponseError(format!(
                        "provider returned {} embeddings for a chunk of {expected} documents",
                        response.embeddings.len(),
                    )));
                }
                Ok::<_, EmbeddingError>((index, response.embeddings, response.usage))
            }
        })
        .buffer_unordered(concurrency.max(1))
        .try_fold(
            (BTreeMap::<usize, Vec<Embedding>>::new(), Usage::new()),
            |(mut acc, mut usage), (index, embeddings, chunk_usage)| async move {
                acc.insert(index, embeddings);
                usage += chunk_usage;
                Ok((acc, usage))
            },
        )
        .await?;

    // Chunks complete out of order; the `BTreeMap` restores input order.
    let flat: Vec<Embedding> = chunked.into_values().flatten().collect();
    let groups = group_batches(&counts, flat)?;

    Ok((documents.into_iter().zip(groups).collect(), usage))
}

/// A fluent accumulator for a document-embedding run — the data-oriented
/// successor to the old `EmbeddingsBuilder`.
///
/// It holds documents and batching knobs and nothing else: **no embedding
/// model, config, runtime, or callback**. The provider is supplied only at the
/// terminal ([`Self::run`] / [`Self::run_with_usage`]), so the job itself stays
/// plain data you can build up, pass around, and reuse.
///
/// `D` is the caller's own document type (payload shape), not a behavior
/// parameter.
///
/// ```no_run
/// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
/// use rig_core::{embeddings::EmbeddingJob, http_runtime::HttpRuntime, providers::openai};
///
/// let cfg = openai::functions::EmbeddingConfig::from_env("text-embedding-3-small")?;
/// let rt = HttpRuntime::new();
///
/// let embedded = EmbeddingJob::new()
///     .documents(["first".to_string(), "second".to_string()])
///     .for_provider(&openai::functions::DESCRIPTOR)
///     .run(|texts| openai::functions::embed(&cfg, &rt, texts))
///     .await?;
/// # let _ = embedded;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct EmbeddingJob<D> {
    documents: Vec<D>,
    max_documents: Option<usize>,
    concurrency: Option<usize>,
}

impl<D> Default for EmbeddingJob<D> {
    fn default() -> Self {
        Self {
            documents: Vec::new(),
            max_documents: None,
            concurrency: None,
        }
    }
}

impl<D> EmbeddingJob<D> {
    /// An empty job.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends one document.
    pub fn document(mut self, document: D) -> Self {
        self.documents.push(document);
        self
    }

    /// Appends documents, preserving order.
    pub fn documents(mut self, documents: impl IntoIterator<Item = D>) -> Self {
        self.documents.extend(documents);
        self
    }

    /// Sets the maximum documents per wire request.
    ///
    /// Left unset, the whole job is sent as one batch.
    pub fn max_documents(mut self, max_documents: usize) -> Self {
        self.max_documents = Some(max_documents);
        self
    }

    /// Sets how many batches are embedded concurrently.
    ///
    /// Left unset, this is [`default_concurrency`] of the effective batch size.
    pub fn concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = Some(concurrency);
        self
    }

    /// Adopts the provider's declared batch limit from its capability sheet.
    ///
    /// A provider that declares no limit leaves the batch size unset (one
    /// batch). Any explicit [`Self::max_documents`] set afterwards wins.
    pub fn for_provider(
        mut self,
        descriptor: &crate::providers::descriptor::ProviderDescriptor,
    ) -> Self {
        self.max_documents = descriptor.max_embedding_documents;
        self
    }

    /// The documents accumulated so far.
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    /// Whether no documents have been added.
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    /// The effective batch size and concurrency this job would run with.
    fn resolved(&self) -> (usize, usize) {
        let max_documents = self.max_documents.unwrap_or(usize::MAX);
        let concurrency = self
            .concurrency
            .unwrap_or_else(|| default_concurrency(max_documents));
        (max_documents, concurrency)
    }
}

impl<D: Embed> EmbeddingJob<D> {
    /// Embeds every document with `embed_batch`, returning each paired with its
    /// embeddings in input order.
    pub async fn run<F, Fut>(
        self,
        embed_batch: F,
    ) -> Result<Vec<(D, OneOrMany<Embedding>)>, EmbeddingError>
    where
        F: Fn(Vec<String>) -> Fut,
        Fut: Future<Output = Result<EmbeddingResponse, EmbeddingError>>,
    {
        let (max_documents, concurrency) = self.resolved();
        embed_documents(self.documents, max_documents, concurrency, embed_batch).await
    }

    /// [`Self::run`], additionally returning token usage summed across chunks.
    pub async fn run_with_usage<F, Fut>(
        self,
        embed_batch: F,
    ) -> Result<(Vec<(D, OneOrMany<Embedding>)>, Usage), EmbeddingError>
    where
        F: Fn(Vec<String>) -> Fut,
        Fut: Future<Output = Result<EmbeddingResponse, EmbeddingError>>,
    {
        let (max_documents, concurrency) = self.resolved();
        embed_documents_with_usage(self.documents, max_documents, concurrency, embed_batch).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn embedding(document: &str) -> Embedding {
        Embedding {
            document: document.to_string(),
            vec: vec![0.0],
        }
    }

    /// Records the batch sizes it was asked to embed, so a test can assert
    /// how a job chunked its documents.
    fn recording_embedder(
        seen: std::sync::Arc<std::sync::Mutex<Vec<usize>>>,
    ) -> impl Fn(Vec<String>) -> std::future::Ready<Result<EmbeddingResponse, EmbeddingError>> {
        move |texts: Vec<String>| {
            seen.lock().expect("batch log").push(texts.len());
            let embeddings = texts.iter().map(|t| embedding(t)).collect();
            std::future::ready(Ok(EmbeddingResponse {
                embeddings,
                usage: Usage {
                    input_tokens: texts.len() as u64,
                    total_tokens: texts.len() as u64,
                    ..Usage::new()
                },
            }))
        }
    }

    #[tokio::test]
    async fn job_matches_the_free_function_and_preserves_input_order() {
        let docs = vec!["a".to_string(), "b".to_string(), "c".to_string()];

        let seen_job = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let via_job = EmbeddingJob::new()
            .documents(docs.clone())
            .max_documents(2)
            .concurrency(1)
            .run(recording_embedder(seen_job.clone()))
            .await
            .expect("job runs");

        let seen_free = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let via_free = embed_documents(docs, 2, 1, recording_embedder(seen_free.clone()))
            .await
            .expect("free function runs");

        let job_docs: Vec<&String> = via_job.iter().map(|(d, _)| d).collect();
        let free_docs: Vec<&String> = via_free.iter().map(|(d, _)| d).collect();
        assert_eq!(job_docs, free_docs);
        assert_eq!(
            job_docs,
            [&"a".to_string(), &"b".to_string(), &"c".to_string()]
        );
        // Identical chunking, not just identical output.
        assert_eq!(
            *seen_job.lock().expect("job batches"),
            *seen_free.lock().expect("free batches")
        );
    }

    #[tokio::test]
    async fn job_reports_the_same_usage_as_the_free_function() {
        let docs = vec!["a".to_string(), "b".to_string()];

        let (_, job_usage) = EmbeddingJob::new()
            .documents(docs.clone())
            .max_documents(1)
            .run_with_usage(recording_embedder(Default::default()))
            .await
            .expect("job runs");
        let (_, free_usage) = embed_documents_with_usage(
            docs,
            1,
            default_concurrency(1),
            recording_embedder(Default::default()),
        )
        .await
        .expect("free function runs");

        assert_eq!(job_usage.total_tokens, free_usage.total_tokens);
        assert_eq!(job_usage.input_tokens, free_usage.input_tokens);
    }

    #[tokio::test]
    async fn unset_batch_size_sends_one_batch_and_for_provider_adopts_the_limit() {
        let docs: Vec<String> = (0..5).map(|i| i.to_string()).collect();

        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        EmbeddingJob::new()
            .documents(docs.clone())
            .run(recording_embedder(seen.clone()))
            .await
            .expect("job runs");
        assert_eq!(*seen.lock().expect("batches"), vec![5]);

        // A descriptor that declares a limit chunks to it.
        let descriptor = crate::providers::descriptor::ProviderDescriptor {
            max_embedding_documents: Some(2),
            ..crate::providers::openai::functions::DESCRIPTOR
        };
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        EmbeddingJob::new()
            .documents(docs)
            .for_provider(&descriptor)
            .concurrency(1)
            .run(recording_embedder(seen.clone()))
            .await
            .expect("job runs");
        assert_eq!(*seen.lock().expect("batches"), vec![2, 2, 1]);
    }

    #[test]
    fn documents_accumulate_in_order_across_both_setters() {
        let job = EmbeddingJob::new()
            .document("a".to_string())
            .documents(["b".to_string(), "c".to_string()])
            .document("d".to_string());

        assert_eq!(job.len(), 4);
        assert!(!job.is_empty());
        assert!(EmbeddingJob::<String>::new().is_empty());
    }

    #[test]
    fn group_batches_is_order_aligned() {
        let groups = group_batches(
            &[2, 1],
            vec![embedding("a"), embedding("b"), embedding("c")],
        )
        .expect("group");
        assert_eq!(groups.len(), 2);
        let first: Vec<_> = groups
            .first()
            .expect("first group")
            .iter()
            .map(|e| e.document.clone())
            .collect();
        assert_eq!(first, ["a", "b"]);
        let second: Vec<_> = groups
            .get(1)
            .expect("second group")
            .iter()
            .map(|e| e.document.clone())
            .collect();
        assert_eq!(second, ["c"]);
    }

    #[test]
    fn group_batches_rejects_empty_batch_and_count_mismatch() {
        assert!(matches!(
            group_batches(&[1, 0], vec![embedding("a")]),
            Err(EmbeddingError::DocumentError(_))
        ));
        assert!(matches!(
            group_batches(&[2], vec![embedding("a")]),
            Err(EmbeddingError::ResponseError(_))
        ));
    }

    #[cfg(feature = "test-utils")]
    #[tokio::test]
    async fn embed_chunked_honors_max_documents_and_preserves_order() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let http_client = crate::test_utils::RecordingHttpClient::new("{}");
        let rt = HttpRuntime::recording(http_client.clone());
        let texts: Vec<String> = (0..5).map(|i| format!("doc-{i}")).collect();
        let calls = AtomicUsize::new(0);

        let response = embed_chunked(
            &rt,
            texts,
            Some(2),
            |chunk| {
                assert!(chunk.len() <= 2);
                http::Request::post("http://embed.test/")
                    .body(serde_json::to_vec(chunk)?)
                    .map_err(|e| EmbeddingError::ResponseError(e.to_string()))
            },
            |_status, _body, documents| {
                let index = calls.fetch_add(1, Ordering::SeqCst);
                let embeddings = documents
                    .into_iter()
                    .map(|document| Embedding {
                        document,
                        vec: vec![index as f64],
                    })
                    .collect();
                let mut usage = Usage::new();
                usage.total_tokens = 1;
                Ok(EmbeddingResponse { embeddings, usage })
            },
        )
        .await
        .expect("embed");

        // 5 documents at max 2 per request → 3 requests, usage summed.
        assert_eq!(http_client.requests().len(), 3);
        assert_eq!(response.usage.total_tokens, 3);
        let documents: Vec<_> = response
            .embeddings
            .iter()
            .map(|e| e.document.clone())
            .collect();
        assert_eq!(documents, ["doc-0", "doc-1", "doc-2", "doc-3", "doc-4"]);
    }

    // ============================================================
    // `embed_documents` — the replacement for `EmbeddingsBuilder`
    // ============================================================

    use crate::embeddings::embed::EmbedError;

    /// A document contributing a single text fragment.
    #[derive(Clone, Debug)]
    struct TextDocument {
        id: String,
        text: String,
    }

    impl TextDocument {
        fn new(id: &str, text: &str) -> Self {
            Self {
                id: id.to_string(),
                text: text.to_string(),
            }
        }
    }

    impl Embed for TextDocument {
        fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
            embedder.embed(self.text.clone());
            Ok(())
        }
    }

    /// A document contributing several text fragments.
    #[derive(Clone, Debug)]
    struct MultiTextDocument {
        id: String,
        texts: Vec<String>,
    }

    impl MultiTextDocument {
        fn new(id: &str, texts: &[&str]) -> Self {
            Self {
                id: id.to_string(),
                texts: texts.iter().map(|t| (*t).to_string()).collect(),
            }
        }
    }

    impl Embed for MultiTextDocument {
        fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
            for text in &self.texts {
                embedder.embed(text.clone());
            }
            Ok(())
        }
    }

    /// A document whose `Embed` implementation fails.
    struct FailingDocument;

    impl Embed for FailingDocument {
        fn embed(&self, _embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
            Err(EmbedError::new(std::io::Error::other("nope")))
        }
    }

    /// A document that contributes no texts at all.
    struct EmptyDocument;

    impl Embed for EmptyDocument {
        fn embed(&self, _embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
            Ok(())
        }
    }

    /// Stand-in for a provider's free `embed`: one canned embedding per input
    /// text, in input order, one token of usage per text.
    async fn canned_embed(texts: Vec<String>) -> Result<EmbeddingResponse, EmbeddingError> {
        let mut usage = Usage::new();
        usage.total_tokens = texts.len() as u64;
        Ok(EmbeddingResponse {
            embeddings: texts
                .into_iter()
                .map(|document| Embedding {
                    document,
                    vec: vec![0.0, 0.1, 0.2],
                })
                .collect(),
            usage,
        })
    }

    fn definitions_multiple_text() -> Vec<MultiTextDocument> {
        vec![
            MultiTextDocument::new(
                "doc0",
                &[
                    "A green alien that lives on cold planets.",
                    "A fictional digital currency that originated in the animated series Rick and Morty.",
                ],
            ),
            MultiTextDocument::new(
                "doc1",
                &[
                    "An ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.",
                    "A fictional creature found in the distant, swampy marshlands of the planet Glibbo in the Andromeda galaxy.",
                ],
            ),
        ]
    }

    fn definitions_multiple_text_2() -> Vec<MultiTextDocument> {
        vec![
            MultiTextDocument::new("doc2", &["Another fake definitions"]),
            MultiTextDocument::new("doc3", &["Some fake definition"]),
        ]
    }

    fn definitions_single_text() -> Vec<TextDocument> {
        vec![
            TextDocument::new("doc0", "A green alien that lives on cold planets."),
            TextDocument::new(
                "doc1",
                "An ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.",
            ),
        ]
    }

    #[tokio::test]
    async fn embed_documents_multiple_text() {
        let result = embed_documents(definitions_multiple_text(), 5, 2, canned_embed)
            .await
            .expect("embed");

        assert_eq!(result.len(), 2);

        let first = result.first().expect("first");
        assert_eq!(first.0.id, "doc0");
        assert_eq!(first.1.len(), 2);
        assert_eq!(
            first.1.first().document,
            "A green alien that lives on cold planets."
        );

        let second = result.get(1).expect("second");
        assert_eq!(second.0.id, "doc1");
        assert_eq!(second.1.len(), 2);
        assert_eq!(
            second.1.rest().first().expect("rest").document,
            "A fictional creature found in the distant, swampy marshlands of the planet Glibbo in the Andromeda galaxy."
        );
    }

    #[tokio::test]
    async fn embed_documents_single_text() {
        let result = embed_documents(definitions_single_text(), 5, 2, canned_embed)
            .await
            .expect("embed");

        assert_eq!(result.len(), 2);

        let first = result.first().expect("first");
        assert_eq!(first.0.id, "doc0");
        assert_eq!(first.1.len(), 1);
        assert_eq!(
            first.1.first().document,
            "A green alien that lives on cold planets."
        );

        let second = result.get(1).expect("second");
        assert_eq!(second.0.id, "doc1");
        assert_eq!(second.1.len(), 1);
        assert_eq!(
            second.1.first().document,
            "An ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land."
        );
    }

    #[tokio::test]
    async fn embed_documents_mixes_single_and_multi_text_documents() {
        let mut documents = definitions_multiple_text();
        documents.extend(definitions_multiple_text_2());

        let result = embed_documents(documents, 5, 2, canned_embed)
            .await
            .expect("embed");

        assert_eq!(result.len(), 4);

        let second = result.get(1).expect("second");
        assert_eq!(second.0.id, "doc1");
        assert_eq!(second.1.len(), 2);
        assert_eq!(
            second.1.first().document,
            "An ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land."
        );

        let third = result.get(2).expect("third");
        assert_eq!(third.0.id, "doc2");
        assert_eq!(third.1.len(), 1);
        assert_eq!(third.1.first().document, "Another fake definitions");
    }

    #[tokio::test]
    async fn embed_documents_accepts_bare_string_collections() {
        let documents: Vec<Vec<String>> = definitions_multiple_text()
            .iter()
            .map(|def| def.texts.clone())
            .collect();

        let result = embed_documents(documents, 5, 2, canned_embed)
            .await
            .expect("embed");

        assert_eq!(result.len(), 2);

        let first = result.first().expect("first");
        assert_eq!(first.1.len(), 2);
        assert_eq!(
            first.1.first().document,
            "A green alien that lives on cold planets."
        );

        let second = result.get(1).expect("second");
        assert_eq!(second.1.len(), 2);
        assert_eq!(
            second.1.rest().first().expect("rest").document,
            "A fictional creature found in the distant, swampy marshlands of the planet Glibbo in the Andromeda galaxy."
        );
    }

    #[tokio::test]
    async fn embed_documents_chunks_by_max_documents_and_preserves_order() {
        use std::sync::Mutex;

        // Two documents of three texts each → 6 texts, chunked by 2 → 3 calls.
        let documents = vec![
            MultiTextDocument::new("doc0", &["a0", "a1", "a2"]),
            MultiTextDocument::new("doc1", &["b0", "b1", "b2"]),
        ];
        let chunks: Mutex<Vec<Vec<String>>> = Mutex::new(Vec::new());

        let (result, usage) = embed_documents_with_usage(documents, 2, 8, |texts: Vec<String>| {
            chunks.lock().expect("lock").push(texts.clone());
            canned_embed(texts)
        })
        .await
        .expect("embed");

        assert_eq!(chunks.lock().expect("lock").len(), 3);
        assert!(
            chunks
                .lock()
                .expect("lock")
                .iter()
                .all(|chunk| chunk.len() == 2)
        );
        assert_eq!(usage.total_tokens, 6);

        let first: Vec<_> = result
            .first()
            .expect("first")
            .1
            .iter()
            .map(|e| e.document.clone())
            .collect();
        assert_eq!(first, ["a0", "a1", "a2"]);
        let second: Vec<_> = result
            .get(1)
            .expect("second")
            .1
            .iter()
            .map(|e| e.document.clone())
            .collect();
        assert_eq!(second, ["b0", "b1", "b2"]);
    }

    #[tokio::test]
    async fn embed_documents_propagates_embed_errors() {
        let result = embed_documents(vec![FailingDocument], 5, 1, canned_embed).await;
        assert!(matches!(result, Err(EmbeddingError::DocumentError(_))));
    }

    #[tokio::test]
    async fn embed_documents_rejects_documents_with_no_texts() {
        let result = embed_documents(vec![EmptyDocument], 5, 1, canned_embed).await;
        assert!(matches!(result, Err(EmbeddingError::DocumentError(_))));
    }

    #[tokio::test]
    async fn embed_documents_propagates_batch_failures() {
        let result = embed_documents(definitions_single_text(), 5, 1, |_texts| async {
            Err::<EmbeddingResponse, _>(EmbeddingError::ProviderError("boom".to_string()))
        })
        .await;
        assert!(matches!(result, Err(EmbeddingError::ProviderError(_))));
    }

    #[tokio::test]
    async fn embed_documents_rejects_short_provider_responses() {
        let result = embed_documents(definitions_single_text(), 5, 1, |_texts| async {
            Ok(EmbeddingResponse {
                embeddings: vec![embedding("only-one")],
                usage: Usage::new(),
            })
        })
        .await;
        assert!(matches!(result, Err(EmbeddingError::ResponseError(_))));
    }

    #[test]
    fn default_concurrency_matches_the_retired_builder() {
        assert_eq!(default_concurrency(1024), 1);
        assert_eq!(default_concurrency(100), 10);
        assert_eq!(default_concurrency(2048), 1);
        assert_eq!(default_concurrency(0), 1024);
    }
}
