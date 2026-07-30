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

#[cfg(test)]
mod tests {
    use super::*;

    fn embedding(document: &str) -> Embedding {
        Embedding {
            document: document.to_string(),
            vec: vec![0.0],
        }
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
