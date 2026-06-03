use httpmock::MockServer;
use rig_core::{
    Embed,
    embeddings::{Embedding, EmbeddingError, EmbeddingModel, EmbeddingsBuilder, embed::{EmbedError, TextEmbedder}},
    vector_store::{InsertDocuments, VectorStoreIndex, request::{Filter, VectorSearchRequest, SearchFilter}},
};
use rig_pinecone::{PineconeClient, PineconeVectorStore};
use serde::{Deserialize, Serialize};

#[derive(Clone)]
struct MockModel;

impl EmbeddingModel for MockModel {
    const MAX_DOCUMENTS: usize = 100;
    type Client = rig_core::client::Nothing;

    fn make(_client: &Self::Client, _model: impl Into<String>, _ndims: Option<usize>) -> Self {
        Self
    }

    fn ndims(&self) -> usize {
        2
    }

    async fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String> + Send,
    ) -> Result<Vec<Embedding>, EmbeddingError> {
        Ok(documents
            .into_iter()
            .map(|d| Embedding {
                document: d,
                vec: vec![0.1, 0.2],
            })
            .collect())
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct MyDoc {
    text: String,
    tag: String,
}

impl Embed for MyDoc {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(self.text.clone());
        Ok(())
    }
}

#[tokio::test]
async fn test_pinecone_upsert_and_query() {
    let server = MockServer::start();

    // 1. Mock the upsert endpoint (matching request without random UUIDs)
    let upsert_mock = server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/vectors/upsert")
            .header("Api-Key", "test-key")
            .header("X-Pinecone-Api-Version", "2025-01");
        then.status(200)
            .json_body(serde_json::json!({
                "upsertedCount": 1
            }));
    });

    // 2. Mock the query endpoint for top_n
    let query_mock = server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/query")
            .header("Api-Key", "test-key")
            .header("X-Pinecone-Api-Version", "2025-01")
            .json_body(serde_json::json!({
                "namespace": "test-namespace",
                "vector": [0.1, 0.2],
                "topK": 2,
                "includeValues": false,
                "includeMetadata": true
            }));
        then.status(200)
            .json_body(serde_json::json!({
                "matches": [
                    {
                        "id": "doc-123",
                        "score": 0.95,
                        "metadata": {
                            "document": {
                                "text": "hello",
                                "tag": "test"
                            }
                        }
                    }
                ]
            }));
    });

    // 3. Mock the query endpoint for top_n_ids
    let query_ids_mock = server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/query")
            .header("Api-Key", "test-key")
            .header("X-Pinecone-Api-Version", "2025-01")
            .json_body(serde_json::json!({
                "namespace": "test-namespace",
                "vector": [0.1, 0.2],
                "topK": 2,
                "includeValues": false,
                "includeMetadata": false
            }));
        then.status(200)
            .json_body(serde_json::json!({
                "matches": [
                    {
                        "id": "doc-123",
                        "score": 0.95
                    }
                ]
            }));
    });

    let client = PineconeClient::new("test-key", &server.url(""));
    let store = PineconeVectorStore::new(client, MockModel, Some("test-namespace".to_string()));

    let docs = vec![MyDoc {
        text: "hello".to_string(),
        tag: "test".to_string(),
    }];

    let documents = EmbeddingsBuilder::new(MockModel)
        .documents(docs)
        .unwrap()
        .build()
        .await
        .unwrap();

    // Test upsert
    store.insert_documents(documents).await.unwrap();
    upsert_mock.assert();

    // Test query (top_n)
    let search_req = VectorSearchRequest::builder()
        .query("hello")
        .samples(2)
        .build();

    let results: Vec<(f64, String, MyDoc)> = store.top_n(search_req).await.unwrap();
    query_mock.assert();

    assert_eq!(results.len(), 1);
    let (score, id, doc) = &results[0];
    assert_eq!(*score, 0.95);
    assert_eq!(id, "doc-123");
    assert_eq!(doc.text, "hello");
    assert_eq!(doc.tag, "test");

    // Test query (top_n_ids)
    let search_req_ids = VectorSearchRequest::builder()
        .query("hello")
        .samples(2)
        .build();

    let id_results = store.top_n_ids(search_req_ids).await.unwrap();
    query_ids_mock.assert();

    assert_eq!(id_results.len(), 1);
    assert_eq!(id_results[0], (0.95, "doc-123".to_string()));
}

#[tokio::test]
async fn test_pinecone_filters() {
    let server = MockServer::start();

    // Mock query with complex filters
    let query_mock = server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/query")
            .header("Api-Key", "test-key")
            .header("X-Pinecone-Api-Version", "2025-01")
            .json_body(serde_json::json!({
                "namespace": "test-namespace",
                "vector": [0.1, 0.2],
                "topK": 2,
                "filter": {
                    "$and": [
                        { "tag": { "$eq": "test" } },
                        { "price": { "$gt": 100 } }
                    ]
                },
                "includeValues": false,
                "includeMetadata": true
            }));
        then.status(200)
            .json_body(serde_json::json!({
                "matches": [
                    {
                        "id": "doc-123",
                        "score": 0.95,
                        "metadata": {
                            "document": {
                                "text": "hello",
                                "tag": "test"
                            }
                        }
                    }
                ]
            }));
    });

    let client = PineconeClient::new("test-key", &server.url(""));
    let store = PineconeVectorStore::new(client, MockModel, Some("test-namespace".to_string()));

    let filter = Filter::and(
        Filter::eq("tag", serde_json::json!("test")),
        Filter::gt("price", serde_json::json!(100)),
    );

    let search_req = VectorSearchRequest::builder()
        .query("hello")
        .samples(2)
        .filter(filter)
        .build();

    let results: Vec<(f64, String, MyDoc)> = store.top_n(search_req).await.unwrap();
    query_mock.assert();

    assert_eq!(results.len(), 1);
}
