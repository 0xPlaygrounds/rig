use crate::completion::Document;
use crate::embeddings::embed::{EmbedError, Embedder};
use crate::embeddings::{Embed, Embedding, EmbeddingResponse, image_document};
use crate::error::ProviderError;
use crate::message::{DocumentSourceKind, Image};

use super::EmbeddingsBuilder;
use crate::driver::{Exchange, Local, Model, Opened, Opening, Step, Transport};
use crate::test_utils::MockEmbeddings;
use crate::wire::Capabilities;

/// The mock embedding wire at a chosen batch size.
fn batches(max_documents: usize) -> Local<crate::operation::Embedding> {
    Local::new("mock").with_capabilities(Capabilities::embedding(max_documents, 10))
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
    fn embed(&self, embedder: &mut Embedder) -> Result<(), EmbedError> {
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
    let error = EmbeddingsBuilder::new(MockEmbeddings::model())
        .document(NTexts::new(0, 0))
        .unwrap()
        .build()
        .await
        .expect_err("a document with no texts has no embeddings");

    assert!(
        matches!(error, ProviderError::Response(_)),
        "unexpected error variant: {error:?}"
    );
    assert!(
        error
            .to_string()
            .contains("document 0 produced nothing to embed"),
        "error should name the offending document: {error}"
    );
}

/// The same, for a document that is not the first — the index in the
/// message has to be the document's own, not a constant.
#[tokio::test]
async fn test_build_names_the_document_that_embeds_no_text() {
    let error = EmbeddingsBuilder::new(MockEmbeddings::model())
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
        error
            .to_string()
            .contains("document 2 produced nothing to embed"),
        "error should name document 2: {error}"
    );
}

/// A model that batches one text at a time, so *every* multi-text document
/// straddles, and answers later texts faster than earlier ones.
#[derive(Clone, Default)]
struct OneAtATimeReversedLatency;

impl Transport<Local<crate::operation::Embedding>> for OneAtATimeReversedLatency {
    fn send(
        &self,
        documents: Vec<String>,
        _exchange: Exchange,
    ) -> Opening<Step<crate::operation::Embedding>> {
        // Earlier texts wait longer, so completion order is close to the
        // reverse of submission order. Texts are named `d{doc}t{i}`, so the
        // position is what follows the last `t`; if that ever stops parsing
        // every batch would wait 0ms, the completion order would stop being
        // inverted, and this test would quietly stop proving anything, hence
        // the refusal rather than `unwrap_or(0)`.
        let Some(position) = documents
            .first()
            .and_then(|text| text.rsplit_once('t'))
            .and_then(|(_, n)| n.parse::<u64>().ok())
        else {
            return Opening::failed(ProviderError::Provider(format!(
                "could not read a text position out of {documents:?}; \
                 this mock cannot invert completion order without it"
            )));
        };
        let delay = 60u64.saturating_sub(position * 10);
        Opening::new(async move {
            tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
            Ok(Opened::new(futures::stream::iter([Ok(Step::End(
                MockEmbeddings::embed(documents),
            ))])))
        })
    }
}

/// Worst case for the span arithmetic: `MAX_DOCUMENTS = 1` means every text
/// is its own batch, all of them run concurrently, and they finish in
/// roughly reverse order. Nothing about the result may depend on that.
#[tokio::test]
async fn test_build_order_survives_one_text_per_batch_finishing_backwards() {
    let doc = NTexts::new(0, 6);
    let expected = doc.expected();

    let result = EmbeddingsBuilder::new(Model::new(batches(1), OneAtATimeReversedLatency))
        .document(doc)
        .unwrap()
        .build()
        .await
        .unwrap();

    assert_eq!(result.len(), 1);
    assert_eq!(returned(&result[0].1), expected);
}

/// The image mock: each image embeds to a one-dimension vector of its byte
/// length, one image to a request.
#[derive(Clone)]
struct ImageLengths;

impl Transport<Local<crate::operation::ImageEmbedding>> for ImageLengths {
    fn send(
        &self,
        images: Vec<Vec<u8>>,
        _exchange: Exchange,
    ) -> Opening<Step<crate::operation::ImageEmbedding>> {
        let response = EmbeddingResponse::new(
            images
                .iter()
                .map(|bytes| Embedding {
                    document: String::new(),
                    vec: vec![bytes.len() as f64],
                })
                .collect(),
        );
        Opening::ready(Opened::new(futures::stream::iter([Ok(Step::End(
            response,
        ))])))
    }
}

/// PNG bytes `len` long.
fn png(len: usize) -> Vec<u8> {
    let mut bytes = b"\x89PNG\r\n\x1a\n".to_vec();
    bytes.resize(len, 0);
    bytes
}

/// A document's text goes to the text model and its images to the image
/// model, and its embeddings come back in the order it embedded them.
#[tokio::test]
async fn a_document_embeds_its_text_and_images_with_their_own_models_in_order() {
    let (small, large) = (png(12), png(20));
    let document = Document::new("cat", "a cat")
        .with_image(Image::from_bytes(&small))
        .with_image(Image::from_bytes(&large));
    let image_model = Model::new(
        Local::new("mock-images").with_capabilities(Capabilities::embedding(1, 1)),
        ImageLengths,
    );

    let built = EmbeddingsBuilder::new(MockEmbeddings::model())
        .image_model(image_model)
        .document(document)
        .expect("inline images embed")
        .build()
        .await
        .expect("both models answer");

    let embeddings = built.first().map(|(_, embeddings)| embeddings.clone());
    let embeddings = embeddings.unwrap_or_default();
    assert_eq!(
        returned(&embeddings),
        vec![
            "a cat".to_owned(),
            image_document(&small),
            image_document(&large)
        ]
    );
    let widths: Vec<Option<f64>> = embeddings
        .iter()
        .skip(1)
        .map(|embedding| embedding.vec.first().copied())
        .collect();
    assert_eq!(widths, vec![Some(12.0), Some(20.0)]);
}

/// An image needs an image model, and a referenced image has no bytes to
/// embed; both are refused before any request.
#[tokio::test]
async fn images_without_an_image_model_or_inline_bytes_are_refused() {
    let photo = Document::new("cat", "").with_image(Image::from_bytes(&png(12)));
    let error = EmbeddingsBuilder::new(MockEmbeddings::model())
        .document(photo)
        .expect("inline images embed")
        .build()
        .await
        .expect_err("no image model");
    assert!(
        error.to_string().contains("document 0 embeds an image"),
        "{error}"
    );

    let linked = Document::new("cat", "").with_image(Image {
        data: DocumentSourceKind::url("https://example.com/cat.png"),
        ..Image::default()
    });
    let refused = EmbeddingsBuilder::new(MockEmbeddings::model()).document(linked);
    assert!(refused.is_err(), "a linked image has no bytes to embed");
}
