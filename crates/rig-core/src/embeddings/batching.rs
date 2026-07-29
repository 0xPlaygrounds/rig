//! Chunking and batch-alignment helpers for the provider `functions`
//! embedding faces.
//!
//! Every provider's free `embed` function honors its descriptor's
//! `max_embedding_documents` by splitting the input into wire-sized chunks
//! and re-assembling one order-aligned [`EmbeddingResponse`]; every
//! `embed_batches` flattens caller batches, embeds once, and regroups the
//! results per batch. Both loops live here so the per-provider modules stay
//! pure request/parse code.

use crate::OneOrMany;
use crate::completion::Usage;
use crate::embeddings::{Embedding, EmbeddingError, EmbeddingResponse};
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
        assert_eq!(
            documents,
            ["doc-0", "doc-1", "doc-2", "doc-3", "doc-4"]
        );
    }
}
