// Test setup failures are bugs in the test itself, so they stop it.
#![allow(clippy::expect_used)]

use rig_core::{
    Embed,
    embeddings::Embedding,
    test_utils::{MockEmbeddings, MockHttpResponse, SequencedHttpClient},
    vector_store::{
        InsertDocuments, VectorStoreError, VectorStoreIndex,
        request::{DynamicSearchFilter, Filter, SearchFilter, VectorSearchRequest},
    },
};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{WeaviateError, WeaviateFilter, WeaviateValue, WeaviateVectorStore};

#[derive(Embed, Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Word {
    id: String,
    #[embed]
    definition: String,
}

fn store(http: &SequencedHttpClient) -> WeaviateVectorStore {
    WeaviateVectorStore::with_http(
        http.clone(),
        MockEmbeddings::model(),
        "http://weaviate.test/",
        "word",
    )
    .expect("`word` is a valid collection name")
}

fn reply(body: serde_json::Value) -> MockHttpResponse {
    MockHttpResponse::success(body.to_string())
}

fn sent_json(http: &SequencedHttpClient, index: usize) -> serde_json::Value {
    let requests = http.requests();
    let request = requests.get(index);
    assert!(request.is_some(), "request {index} was sent");
    request
        .and_then(|request| serde_json::from_slice(&request.body).ok())
        .unwrap_or_default()
}

#[tokio::test]
async fn top_n_sends_one_graphql_query_and_scores_by_similarity() {
    let http = SequencedHttpClient::new([reply(json!({
        "data": { "Get": { "Word": [
            {
                "document": "{\"id\":\"doc1\",\"definition\":\"a green alien\"}",
                "_additional": { "id": "6f1c0c8e-0000-4000-8000-000000000001", "distance": 0.25 }
            }
        ] } }
    }))]);
    let store = store(&http).api_key("secret").target_vector("body");
    let req = VectorSearchRequest::builder()
        .query("What is a flurbo?")
        .samples(3)
        .threshold(0.5)
        .filter(
            WeaviateFilter::eq("lang", "en".into()).and(WeaviateFilter::gt("year", 2020.into())),
        )
        .build();

    let results = store.top_n::<Word>(req).await.expect("search succeeds");

    assert_eq!(results.len(), 1);
    let hit = results.first();
    assert_eq!(hit.map(|hit| hit.score), Some(0.75));
    assert_eq!(
        hit.map(|hit| hit.document.clone()),
        Some(Word {
            id: "doc1".into(),
            definition: "a green alien".into(),
        })
    );

    let requests = http.requests();
    let request = requests.first();
    assert_eq!(
        request.map(|request| request.uri.as_str()),
        Some("http://weaviate.test/v1/graphql")
    );
    assert_eq!(
        request.and_then(|request| request.headers.get("authorization")),
        Some(&rig_core::http_client::HeaderValue::from_static(
            "Bearer secret"
        ))
    );
    assert_eq!(
        sent_json(&http, 0),
        json!({
            "query": "{ Get { Word(nearVector: {vector: [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], \
                distance: 0.5, targetVectors: [\"body\"]}, limit: 3, \
                where: {operator: And, operands: [\
                {path: [\"lang\"], operator: Equal, valueText: \"en\"}, \
                {path: [\"year\"], operator: GreaterThan, valueInt: 2020}]}) \
                { document _additional { id distance } } } }"
        })
    );
}

#[tokio::test]
async fn top_n_ids_selects_no_document_and_graphql_errors_fail() {
    let http = SequencedHttpClient::new([
        reply(json!({
            "data": { "Get": { "Word": [
                { "_additional": { "id": "a", "distance": 0.0 } },
                { "_additional": { "id": "b", "distance": 1.5 } }
            ] } }
        })),
        reply(json!({
            "data": { "Get": { "Word": null } },
            "errors": [{ "message": "no such class" }]
        })),
    ]);
    let store = store(&http);
    let req = VectorSearchRequest::builder().query("q").samples(2).build();

    let ids = store.top_n_ids(req.clone()).await.expect("search succeeds");
    let ids: Vec<_> = ids.into_iter().map(|hit| (hit.id, hit.score)).collect();
    assert_eq!(ids, [("a".to_owned(), 1.0), ("b".to_owned(), -0.5)]);
    let query = sent_json(&http, 0);
    assert_eq!(
        query.get("query").and_then(|query| query.as_str()),
        Some(
            "{ Get { Word(nearVector: {vector: [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}, \
             limit: 2) { _additional { id distance } } } }"
        )
    );

    let failed = store.top_n_ids(req).await;
    assert!(
        matches!(&failed, Err(VectorStoreError::DatastoreError(error)) if error.to_string().contains("no such class")),
        "{failed:?}"
    );
}

#[tokio::test]
async fn insert_documents_batches_objects_and_reports_rejections() {
    let http = SequencedHttpClient::new([reply(json!([
        { "id": "a", "result": {} },
        { "id": "b", "result": { "errors": { "error": [{ "message": "vector dimensions do not match" }] } } }
    ]))]);
    let store = store(&http);
    let embedding = |text: &str| Embedding {
        document: text.into(),
        vec: vec![1.0, 0.0],
    };
    let documents = vec![
        (
            Word {
                id: "doc1".into(),
                definition: "one".into(),
            },
            vec![embedding("one")],
        ),
        (
            Word {
                id: "doc2".into(),
                definition: "two".into(),
            },
            vec![embedding("two")],
        ),
    ];

    let inserted = store.insert_documents(documents).await;

    assert!(
        matches!(&inserted, Err(VectorStoreError::DatastoreError(error)) if error.to_string().contains("vector dimensions do not match")),
        "{inserted:?}"
    );
    assert_eq!(
        http.requests().first().map(|request| request.uri.clone()),
        Some("http://weaviate.test/v1/batch/objects".to_owned())
    );
    assert_eq!(
        sent_json(&http, 0),
        json!({ "objects": [
            {
                "class": "Word",
                "properties": { "document": "{\"id\":\"doc1\",\"definition\":\"one\"}" },
                "vector": [1.0, 0.0]
            },
            {
                "class": "Word",
                "properties": { "document": "{\"id\":\"doc2\",\"definition\":\"two\"}" },
                "vector": [1.0, 0.0]
            }
        ] })
    );
}

#[test]
fn filters_escape_their_input_and_check_value_types() {
    let filter = WeaviateFilter::like("ti\"tle", "*\"}) { x *")
        .or(WeaviateFilter::contains_any(
            "tags",
            vec!["a".into(), "b".into()],
        ))
        .not();
    assert_eq!(
        filter.to_graphql().ok().as_deref(),
        Some(
            "{operator: Not, operands: [{operator: Or, operands: [\
             {path: [\"ti\\\"tle\"], operator: Like, valueText: \"*\\\"}) { x *\"}, \
             {path: [\"tags\"], operator: ContainsAny, valueText: [\"a\", \"b\"]}]}]}"
        )
    );

    let mixed = WeaviateFilter::contains_all("tags", vec!["a".into(), 1.into()]);
    assert!(mixed.to_graphql().is_err());
    assert!(
        WeaviateFilter::eq("score", f64::NAN.into())
            .to_graphql()
            .is_err()
    );

    let dynamic = WeaviateFilter::from_dynamic_filter(
        Filter::eq("count", json!(3)).and(Filter::lt("ratio", json!(0.5))),
    );
    assert_eq!(
        dynamic
            .ok()
            .and_then(|filter| filter.to_graphql().ok())
            .as_deref(),
        Some(
            "{operator: And, operands: [\
             {path: [\"count\"], operator: Equal, valueInt: 3}, \
             {path: [\"ratio\"], operator: LessThan, valueNumber: 0.5}]}"
        )
    );
    assert!(WeaviateFilter::from_dynamic_filter(Filter::eq("x", json!(null))).is_err());
    assert_eq!(
        WeaviateValue::try_from(json!("x")).ok(),
        Some(WeaviateValue::Text("x".into()))
    );
}

#[test]
fn collection_names_are_capitalized_and_validated() {
    let http = SequencedHttpClient::new([]);
    assert_eq!(store(&http).collection(), "Word");
    for name in ["", "1word", "Word{", "Wo rd"] {
        let built =
            WeaviateVectorStore::with_http(http.clone(), MockEmbeddings::model(), "http://x", name);
        assert!(
            matches!(built, Err(WeaviateError::InvalidCollectionName(_))),
            "{name:?} was accepted"
        );
    }
}
