use super::*;
use crate::driver::Bound;
use crate::embeddings::EmbeddingModel as _;
use crate::rerank::RerankModel as _;
use crate::test_utils::RecordingHttpClient;
use crate::wire::secret::tests::a_config_reloads_without_its_credential;

fn voyage() -> VoyageAi {
    VoyageAi::new("voyage-test-key")
}

fn body_of(encoded: &Encoded) -> serde_json::Value {
    let [request] = encoded.requests.as_slice() else {
        panic!(
            "expected exactly one request, got {}",
            encoded.requests.len()
        );
    };
    match request.body() {
        Body::Bytes(bytes) => serde_json::from_slice(bytes).expect("the body is JSON"),
        Body::Multipart(_) => panic!("neither Voyage wire sends a multipart body"),
    }
}

/// `POST /embeddings`' reply shape, with two-element vectors in place of the
/// 1024-element ones the models return.
const EMBED_BODY: &str = r#"{"object":"list","data":[{"object":"embedding","embedding":[0.5,-0.25],"index":0},{"object":"embedding","embedding":[0.125,0.0],"index":1}],"model":"voyage-3.5","usage":{"total_tokens":9}}"#;

#[tokio::test]
async fn an_embedding_reply_pairs_its_vectors_with_the_texts_that_were_sent() {
    let response = Bound::new(
        voyage().embeddings("voyage-3.5", None),
        RecordingHttpClient::new(EMBED_BODY),
    )
    .embed_texts_response(vec!["first".to_owned(), "second".to_owned()])
    .await
    .expect("the reply decodes");

    assert_eq!(
        response
            .embeddings
            .iter()
            .map(|embedding| (embedding.document.as_str(), embedding.vec.as_slice()))
            .collect::<Vec<_>>(),
        vec![
            ("first", [0.5, -0.25].as_slice()),
            ("second", [0.125, 0.0].as_slice()),
        ]
    );
    assert_eq!(response.model.as_deref(), Some("voyage-3.5"));
    // Voyage reports one counter; every token of an embedding is input.
    assert_eq!(response.usage.input_tokens, Some(9));
    assert_eq!(response.usage.total_tokens, Some(9));
    assert_eq!(response.usage.output_tokens, None);
}

/// Voyage's server defaults are "field absent", so an unset option must not
/// be sent — sending `input_type: null` is not the same request.
#[test]
fn an_unset_option_is_absent_from_the_request() {
    let encoded = voyage()
        .embeddings("voyage-3.5", None)
        .encode(vec!["first".to_owned()], Mode::Unary)
        .expect("the request encodes");

    assert_eq!(
        body_of(&encoded),
        serde_json::json!({ "model": "voyage-3.5", "input": ["first"] })
    );

    let encoded = voyage()
        .embeddings("voyage-3.5", None)
        .with_input_type("query")
        .with_truncation(false)
        .with_output_dimension(256)
        .encode(vec!["first".to_owned()], Mode::Unary)
        .expect("the request encodes");

    assert_eq!(
        body_of(&encoded),
        serde_json::json!({
            "model": "voyage-3.5",
            "input": ["first"],
            "input_type": "query",
            "truncation": false,
            "output_dimension": 256,
        })
    );
}

#[test]
fn an_embedding_wire_reports_the_width_it_asked_for() {
    assert_eq!(
        voyage().embeddings("voyage-3.5", None).capabilities(),
        EmbeddingCapabilities::new(1024, 1024)
    );
    assert_eq!(
        voyage()
            .embeddings("voyage-3.5", None)
            .with_output_dimension(256)
            .capabilities(),
        EmbeddingCapabilities::new(1024, 256),
        "a vector store sizes its index from `ndims`, so asking Voyage for a \
         narrower vector must change what the wire reports"
    );
}

/// `POST /rerank`'s reply shape: scores in relevance order, each naming the
/// index of the document it scored.
const RERANK_BODY: &str = r#"{"object":"list","data":[{"relevance_score":0.9,"index":1},{"relevance_score":0.1,"index":0}],"model":"rerank-2.5","usage":{"total_tokens":26}}"#;

#[tokio::test]
async fn a_rerank_reply_keeps_the_provider_order_and_the_indices_it_named() {
    let response = Bound::new(
        voyage().rerank("rerank-2.5"),
        RecordingHttpClient::new(RERANK_BODY),
    )
    .rerank(
        "which is best?",
        vec!["worse".to_owned(), "better".to_owned()],
    )
    .await
    .expect("the reply decodes");

    assert_eq!(
        response
            .results
            .iter()
            .map(|result| (result.index, result.relevance_score))
            .collect::<Vec<_>>(),
        vec![(1, 0.9), (0, 0.1)]
    );
    assert_eq!(response.model.as_deref(), Some("rerank-2.5"));
    assert_eq!(response.usage.input_tokens, Some(26));
    assert_eq!(response.usage.total_tokens, Some(26));
}

#[test]
fn a_rerank_request_carries_the_query_the_documents_and_the_options() {
    let encoded = voyage()
        .rerank("rerank-2.5")
        .with_top_k(1)
        .with_return_documents(true)
        .encode(
            RerankRequest {
                query: "which is best?".to_owned(),
                documents: vec!["worse".to_owned(), "better".to_owned()],
            },
            Mode::Unary,
        )
        .expect("the request encodes");

    assert_eq!(
        body_of(&encoded),
        serde_json::json!({
            "query": "which is best?",
            "documents": ["worse", "better"],
            "model": "rerank-2.5",
            "top_k": 1,
            "return_documents": true,
        })
    );
}

/// The batch limit is the wire's capability, and it is what
/// `RerankModel::max_documents` reports.
#[test]
fn a_rerank_wire_declares_the_batch_limit() {
    let bound = Bound::new(
        voyage().rerank("rerank-2.5"),
        RecordingHttpClient::new(RERANK_BODY),
    );
    assert_eq!(voyage().rerank("rerank-2.5").capabilities(), 1000);
    assert_eq!(bound.max_documents(), 1000);
}

#[test]
fn a_serialized_config_carries_no_key_material() {
    a_config_reloads_without_its_credential(
        &VoyageAi::new("voyage-test-key"),
        "voyage-test-key",
        |voyage| &voyage.api_key,
    );

    for wire in [
        serde_json::to_string(&voyage().embeddings("voyage-3.5", None)),
        serde_json::to_string(&voyage().rerank("rerank-2.5")),
    ] {
        let serialized = wire.expect("the wire serializes");
        assert!(
            !serialized.contains("voyage-test-key"),
            "a wire a host may persist must not carry the credential: {serialized}"
        );
    }
}
