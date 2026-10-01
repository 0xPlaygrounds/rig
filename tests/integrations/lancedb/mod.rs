use serde_json::json;

use crate::common::{mock_embeddings, openai_client};
use fixture::{FLUMBUZZLE, as_record_batch, flumbuzzles, words};
use lancedb::index::vector::IvfPqIndexBuilder;
use rig::lancedb::{LanceDbVectorIndex, SearchParams};
use rig::vector_store::VectorSearchResult;
use rig::{
    driver::Model, embeddings::EmbeddingsBuilder, prelude::*, providers::openai,
    vector_store::VectorStoreIndex,
};

#[path = "./fixtures/lib.rs"]
mod fixture;

#[tokio::test]
async fn vector_search_test() {
    // Setup mock openai API
    let server = httpmock::MockServer::start();
    mock_embeddings_api(&server);

    let openai_client = openai_client(&server);

    // Select an embedding model.
    let model = openai_client.embedding(openai::TEXT_EMBEDDING_ADA_002, None);

    // Initialize LanceDB locally.
    let store = assert_fs::TempDir::new().unwrap();
    let db = lancedb::connect(store.path().to_str().unwrap())
        .execute()
        .await
        .unwrap();

    let table_name = "definitions";
    let (vector_store_index, _) = index_definitions(&db, model, table_name).await;

    let query = "My boss says I zindle too much, what does that mean?";
    let req = VectorSearchRequest::builder()
        .query(query)
        .samples(1)
        .build();

    // Query the index
    let results = vector_store_index
        .top_n::<serde_json::Value>(req)
        .await
        .unwrap();

    let VectorSearchResult {
        score: distance,
        document: value,
        ..
    } = &results.first().unwrap();

    assert_eq!(
        *value,
        json!({
            "_distance": distance,
            "definition": "Definition of *zindle (verb)*: to pretend to be working on something important while actually doing something completely unrelated or unproductive.",
            "id": "doc1"
        })
    );

    db.drop_table(table_name, &[]).await.unwrap();
}

#[tokio::test]
async fn agent_with_dynamic_context_test() {
    // Setup mock openai API
    let server = httpmock::MockServer::start();
    mock_embeddings_api(&server);

    // Mock completions API for agent response
    server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path_includes("/chat/completions");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "To \"zindle\" means to pretend to be working on something important while actually doing something completely unrelated or unproductive."
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150
                }
            }));
    });

    // Initialize OpenAI client
    let openai_client = openai::OpenAIConfig::new("TEST")
        .with_base_url(server.base_url())
        // The mock answers Chat Completions, not the Responses default.
        .with_route(openai::Route::Chat)
        .client();

    // Select an embedding model.
    let model = openai_client.embedding(openai::TEXT_EMBEDDING_ADA_002, None);

    // Initialize LanceDB locally.
    let store = assert_fs::TempDir::new().unwrap();
    let db = lancedb::connect(store.path().to_str().unwrap())
        .execute()
        .await
        .unwrap();

    let table_name = "agent_definitions";
    let (vector_store_index, top_k) = index_definitions(&db, model, table_name).await;

    // Build RAG agent with dynamic context.
    let agent = AgentBuilder::new(openai_client.completion(openai::GPT_4O))
        .dynamic_context(top_k, vector_store_index)
        .build();

    let query = "My boss says I zindle too much, what does that mean?";

    let response = agent.prompt(query).await.unwrap().output;

    assert!(response.contains("zindle") || response.contains("pretend to be working"));
    assert!(response.contains("important") || response.contains("unproductive"));

    db.drop_table(table_name, &[]).await.unwrap();
}

/// Mocks the embeddings of [`words`] followed by [`flumbuzzles`], and of the
/// `zindle` query, which lands nearest `doc1`.
fn mock_embeddings_api(server: &httpmock::MockServer) {
    let mut inputs: Vec<String> = words().into_iter().map(|word| word.definition).collect();
    inputs.extend(std::iter::repeat_n(FLUMBUZZLE.to_string(), 256));
    let mut embeddings = vec![vec![0.1; 1536], vec![0.0023064255; 1536], vec![0.2; 1536]];
    embeddings.extend(std::iter::repeat_n(vec![0.2; 1536], 256));
    mock_embeddings(
        server,
        json!({ "input": inputs, "model": "text-embedding-ada-002" }),
        embeddings,
    );
    mock_embeddings(
        server,
        json!({
            "input": ["My boss says I zindle too much, what does that mean?"],
            "model": "text-embedding-ada-002",
        }),
        [vec![0.0023064254; 1536]],
    );
}

/// Embeds [`words`] and [`flumbuzzles`] into a new `table_name` table, builds
/// its IVF-PQ index, and returns the vector index with its row count.
async fn index_definitions(
    db: &lancedb::Connection,
    model: Model<openai::wire::Embeddings>,
    table_name: &str,
) -> (LanceDbVectorIndex, usize) {
    let embeddings = EmbeddingsBuilder::new(model.clone())
        .documents(words())
        .unwrap()
        .documents(flumbuzzles())
        .unwrap()
        .build()
        .await
        .unwrap();
    let rows = embeddings.len();

    let batch = as_record_batch(embeddings, model.capabilities().ndims).unwrap();
    let table = db
        .create_table(table_name, vec![batch])
        .execute()
        .await
        .unwrap();

    // See [LanceDB indexing](https://lancedb.github.io/lancedb/concepts/index_ivfpq/#product-quantization) for more information
    table
        .create_index(
            &["embedding"],
            lancedb::index::Index::IvfPq(IvfPqIndexBuilder::default()),
        )
        .execute()
        .await
        .unwrap();

    let index = LanceDbVectorIndex::new(table, model, "id", SearchParams::default())
        .await
        .unwrap();
    (index, rows)
}
