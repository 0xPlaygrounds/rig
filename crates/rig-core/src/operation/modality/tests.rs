use serde_json::json;

use super::*;
use crate::driver::Local;
use crate::embeddings::EmbeddingResponse;
use crate::rerank::RerankResponse;
use crate::wire::Mode;

/// A reply as the driver saw it: its wire's name, its whole body and the
/// transport's request id.
fn reply() -> Reply {
    Reply {
        provider: "wire".to_owned(),
        raw: json!({"id": "body", "unmodeled": true}),
        provider_request_id: Some("req-1".to_owned()),
    }
}

/// A rerank response whose decoder wrote the driver's facts itself, and
/// reported its ids empty.
fn decoded_rerank() -> RerankResponse {
    RerankResponse {
        provider: "decoder".to_owned(),
        provider_request_id: Some("decoder-req".to_owned()),
        raw: json!({"re": "serialized"}),
        model: Some(String::new()),
        response_id: Some(String::new()),
        ..RerankResponse::new(Vec::new())
    }
}

fn rerank(response: RerankResponse, reply: Reply) -> RerankResponse {
    let request = RerankRequest {
        query: "q".to_owned(),
        documents: Vec::new(),
    };
    crate::test_utils::fold_for(&request, &Local::<Rerank>::new("wire"), Mode::Unary)
        .finish(response, reply)
        .expect("the reply folds")
}

#[test]
fn the_driver_writes_its_facts_over_what_a_decoder_wrote() {
    let response = rerank(decoded_rerank(), reply());

    assert_eq!(response.provider, "wire");
    assert_eq!(response.provider_request_id.as_deref(), Some("req-1"));
    assert_eq!(response.raw, reply().raw);
    assert_eq!(response.model, None, "an empty model is no model");
    assert_eq!(response.response_id, None, "an empty id is no id");
}

#[test]
fn a_fact_the_transport_did_not_report_stays_unset() {
    let silent = Reply {
        raw: serde_json::Value::Null,
        provider_request_id: None,
        ..reply()
    };
    let response = rerank(decoded_rerank(), silent);

    assert_eq!(response.provider_request_id, None);
    assert_eq!(response.raw, serde_json::Value::Null);
}

#[test]
fn an_empty_request_id_the_reply_carries_is_no_id() {
    let empty = Reply {
        provider_request_id: Some(String::new()),
        ..reply()
    };
    let response = rerank(decoded_rerank(), empty);

    assert_eq!(response.provider_request_id, None);
}

#[test]
fn an_embedding_fold_writes_the_drivers_facts() {
    let fold = crate::test_utils::fold_for(
        &vec!["doc".to_owned()],
        &Local::<Embedding>::new("wire"),
        Mode::Unary,
    );
    let decoded = EmbeddingResponse {
        provider: "decoder".to_owned(),
        model: Some(String::new()),
        ..EmbeddingResponse::new(vec![Vector {
            document: String::new(),
            vec: vec![0.5],
        }])
    };
    let response = fold.finish(decoded, reply()).expect("the reply folds");

    assert_eq!(response.provider, "wire");
    assert_eq!(response.provider_request_id.as_deref(), Some("req-1"));
    assert_eq!(response.raw, reply().raw);
    assert_eq!(response.model, None);
    assert_eq!(response.embeddings[0].document, "doc");
}
