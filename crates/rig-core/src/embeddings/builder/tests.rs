use crate::embeddings::embed::{EmbedError, TextEmbedder};
use crate::embeddings::{Embed, Embedding, EmbeddingError, EmbeddingModel, EmbeddingResponse};
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
    );
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
    );
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
    );
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
    );
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
    fn max_documents(&self) -> usize {
        5
    }

    fn ndims(&self) -> usize {
        10
    }

    async fn embed_texts_response(
        &self,
        documents: impl IntoIterator<Item = String> + crate::wasm_compat::WasmCompatSend,
    ) -> Result<EmbeddingResponse, EmbeddingError> {
        let nth = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if nth == 0 {
            tokio::time::sleep(std::time::Duration::from_millis(150)).await;
        }
        Ok(EmbeddingResponse::new(
            documents
                .into_iter()
                .map(|document| Embedding {
                    document,
                    vec: vec![0.0; 10],
                })
                .collect(),
            "mock",
        ))
    }
}

/// A model whose batches finish in reverse submission order: batch `n`
/// sleeps longer the earlier it was submitted.
///
/// `SlowFirstBatchModel` only inverts the *first* boundary, so a document
/// straddling a later one still comes back correct even unfixed. This
/// inverts every boundary.
#[derive(Clone)]
struct DescendingLatencyModel {
    batches: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl DescendingLatencyModel {
    fn new() -> Self {
        Self {
            batches: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }
}

impl EmbeddingModel for DescendingLatencyModel {
    fn max_documents(&self) -> usize {
        5
    }

    fn ndims(&self) -> usize {
        10
    }

    async fn embed_texts_response(
        &self,
        documents: impl IntoIterator<Item = String> + crate::wasm_compat::WasmCompatSend,
    ) -> Result<EmbeddingResponse, EmbeddingError> {
        let nth = self
            .batches
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst) as u64;
        tokio::time::sleep(std::time::Duration::from_millis(
            120u64.saturating_sub(nth * 40),
        ))
        .await;
        Ok(EmbeddingResponse::new(
            documents
                .into_iter()
                .map(|document| Embedding {
                    document,
                    vec: vec![0.0; 10],
                })
                .collect(),
            "mock",
        ))
    }
}

/// A document contributing `n` texts, each naming **both** its owner and
/// its position: `d{doc}t0 .. d{doc}t{n-1}`.
///
/// The owner half is load-bearing. Documents of the same length would
/// otherwise produce byte-identical text lists, and a test asserting
/// `["t0", "t1", "t2"]` per document could not tell a document's own run
/// from a neighbour's — it would pass even if every document were handed
/// the next one's embeddings wholesale.
#[derive(Debug)]
struct NTexts {
    doc: usize,
    n: usize,
}

impl NTexts {
    fn new(doc: usize, n: usize) -> Self {
        Self { doc, n }
    }

    /// The texts this document is expected to get back, in order.
    fn expected(&self) -> Vec<String> {
        (0..self.n).map(|i| format!("d{}t{i}", self.doc)).collect()
    }
}

impl Embed for NTexts {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        for i in 0..self.n {
            embedder.embed(format!("d{}t{i}", self.doc));
        }
        Ok(())
    }
}

/// The texts a document actually got back, in order.
fn returned(embeddings: &[Embedding]) -> Vec<String> {
    embeddings
        .iter()
        .map(|embedding| embedding.document.clone())
        .collect()
}

/// rig#2345 — a document whose texts straddle a `MAX_DOCUMENTS` boundary
/// must get its embeddings back in text order.
///
/// Six texts against a limit of 5 splits into `[d0t0..d0t4]` and `[d0t5]`;
/// the delayed first batch makes the trailing one finish first. Before the
/// slot index this returned the sixth text's embedding at index 0.
#[tokio::test]
async fn test_build_preserves_text_order_within_a_straddling_document() {
    let doc = NTexts::new(0, 6);
    let expected = doc.expected();

    let result = EmbeddingsBuilder::new(SlowFirstBatchModel::new())
        .document(doc)
        .unwrap()
        .build()
        .await
        .unwrap();

    assert_eq!(result.len(), 1);
    assert_eq!(returned(&result[0].1), expected);
}

/// The same guarantee with **more than one** straddle inverted at once:
/// every document's texts land in its own list, in order, none borrowed
/// from a neighbour. Texts carry their owner, so "borrowed from a
/// neighbour" is something this can actually observe.
///
/// 4 documents x 3 texts = 12 slots over a limit of 5 gives batches
/// `[0,5) [5,10) [10,12)`, so doc1 (slots 3..6) and doc3 (slots 9..12) each
/// straddle. `DescendingLatencyModel` inverts both boundaries — with a
/// model that only delays the first batch, doc3's two batches still arrive
/// in submission order and it comes back correct even unfixed.
#[tokio::test]
async fn test_build_preserves_text_order_across_many_straddling_documents() {
    let docs: Vec<NTexts> = (0..4).map(|doc| NTexts::new(doc, 3)).collect();
    let expected: Vec<Vec<String>> = docs.iter().map(NTexts::expected).collect();

    let result = EmbeddingsBuilder::new(DescendingLatencyModel::new())
        .documents(docs)
        .unwrap()
        .build()
        .await
        .unwrap();

    assert_eq!(result.len(), 4);
    for (index, (_, embeddings)) in result.iter().enumerate() {
        assert_eq!(
            returned(embeddings),
            expected[index],
            "document {index} did not get its own texts, in order"
        );
    }
}

/// A document that embeds no text has no embeddings to return. This has
/// always been an error rather than an empty list, and the slot rewrite
/// keeps it that way — that behavior is what this pins, and it holds on
/// both sides of the fix.
///
/// The wording changed: the message now names the document and says what
/// caused it, where before it was the unlocated `"missing embedding for
/// document after batch merge"`. Only the second assertion below is new
/// behavior.
#[tokio::test]
async fn test_build_rejects_a_document_that_embeds_no_text() {
    let error = EmbeddingsBuilder::new(MockEmbeddingModel)
        .document(NTexts::new(0, 0))
        .unwrap()
        .build()
        .await
        .expect_err("a document with no texts has no embeddings");

    assert!(
        matches!(error, EmbeddingError::ResponseError(_)),
        "unexpected error variant: {error:?}"
    );
    assert!(
        error.to_string().contains("document 0 produced no text"),
        "error should name the offending document: {error}"
    );
}

/// The same, for a document that is not the first — the index in the
/// message has to be the document's own, not a constant.
#[tokio::test]
async fn test_build_names_the_document_that_embeds_no_text() {
    let error = EmbeddingsBuilder::new(MockEmbeddingModel)
        .documents(vec![
            NTexts::new(0, 2),
            NTexts::new(1, 2),
            NTexts::new(2, 0),
        ])
        .unwrap()
        .build()
        .await
        .expect_err("a document with no texts has no embeddings");

    assert!(
        error.to_string().contains("document 2 produced no text"),
        "error should name document 2: {error}"
    );
}

/// A model that batches one text at a time, so *every* multi-text document
/// straddles, and answers later texts faster than earlier ones.
#[derive(Clone, Default)]
struct OneAtATimeReversedLatency;

impl EmbeddingModel for OneAtATimeReversedLatency {
    fn max_documents(&self) -> usize {
        1
    }

    fn ndims(&self) -> usize {
        10
    }

    async fn embed_texts_response(
        &self,
        documents: impl IntoIterator<Item = String> + crate::wasm_compat::WasmCompatSend,
    ) -> Result<EmbeddingResponse, EmbeddingError> {
        let documents: Vec<String> = documents.into_iter().collect();
        // Earlier texts wait longer, so completion order is close to the
        // reverse of submission order. Texts are named `d{doc}t{i}`, so the
        // position is what follows the last `t`; if that ever stops
        // parsing every batch waits 0ms, the completion order stops being
        // inverted, and this test quietly stops proving anything — hence
        // the assert rather than `unwrap_or(0)`.
        let position = documents
            .first()
            .and_then(|text| text.rsplit_once('t'))
            .and_then(|(_, n)| n.parse::<u64>().ok());
        assert!(
            position.is_some(),
            "could not read a text position out of {documents:?}; \
                 this mock cannot invert completion order without it"
        );
        let delay = position.map_or(0, |n| 60u64.saturating_sub(n * 10));
        tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
        Ok(EmbeddingResponse::new(
            documents
                .into_iter()
                .map(|document| Embedding {
                    document,
                    vec: vec![0.0; 10],
                })
                .collect(),
            "mock",
        ))
    }
}

/// Worst case for the span arithmetic: `MAX_DOCUMENTS = 1` means every text
/// is its own batch, all of them run concurrently, and they finish in
/// roughly reverse order. Nothing about the result may depend on that.
#[tokio::test]
async fn test_build_order_survives_one_text_per_batch_finishing_backwards() {
    let doc = NTexts::new(0, 6);
    let expected = doc.expected();

    let result = EmbeddingsBuilder::new(OneAtATimeReversedLatency)
        .document(doc)
        .unwrap()
        .build()
        .await
        .unwrap();

    assert_eq!(result.len(), 1);
    assert_eq!(returned(&result[0].1), expected);
}

/// Documents that tile the batch size exactly, so every document boundary
/// is also a batch boundary — the case where an off-by-one in the span
/// arithmetic would hand a document its neighbour's run.
#[tokio::test]
async fn test_build_order_when_documents_tile_the_batch_size_exactly() {
    // 3 documents x 5 texts, MAX_DOCUMENTS = 5: batches align exactly with
    // document boundaries.
    let docs: Vec<NTexts> = (0..3).map(|doc| NTexts::new(doc, 5)).collect();
    let expected: Vec<Vec<String>> = docs.iter().map(NTexts::expected).collect();

    let result = EmbeddingsBuilder::new(SlowFirstBatchModel::new())
        .documents(docs)
        .unwrap()
        .build()
        .await
        .unwrap();

    assert_eq!(result.len(), 3);
    for (index, (_, embeddings)) in result.iter().enumerate() {
        assert_eq!(
            returned(embeddings),
            expected[index],
            "document {index} did not get its own run"
        );
    }
}
