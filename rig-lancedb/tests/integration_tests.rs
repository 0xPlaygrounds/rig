use serde_json::json;

use arrow_array::RecordBatchIterator;
use fixture::{Word, as_record_batch, schema, words};
use lancedb::index::vector::IvfPqIndexBuilder;
use rig::{
    client::EmbeddingsClient,
    embeddings::{EmbeddingModel, EmbeddingsBuilder},
    providers::openai,
    vector_store::{VectorStoreIndex, request::VectorSearchRequest},
};
use rig_lancedb::{LanceDbVectorIndex, SearchParams};
use std::sync::Arc;

#[path = "./fixtures/lib.rs"]
mod fixture;

#[tokio::test]
async fn vector_search_test() {
    // Setup mock openai API
    let server = httpmock::MockServer::start();

    server.mock(|when, then| {
        let mut req_data = vec![
            "Definition of *flumbrel (noun)*: a small, seemingly insignificant item that you constantly lose or misplace, such as a pen, hair tie, or remote control.",
            "Definition of *zindle (verb)*: to pretend to be working on something important while actually doing something completely unrelated or unproductive.",
            "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.",
        ];
        req_data.append(vec!["Definition of *flumbuzzle (noun)*: A sudden, inexplicable urge to rearrange or reorganize small objects, such as desk items or books, for no apparent reason."; 256].as_mut());

        when.method(httpmock::Method::POST)
            .path("/embeddings")
            .header("Authorization", "Bearer TEST")
            .header("Content-Type", "application/json")
            .json_body(json!({
                "input": req_data,
                "model": "text-embedding-ada-002",
                "dimensions": 1536
            }));

        let mut resp_data = vec![
            json!({
                "object": "embedding",
                "embedding": vec![0.1; 1536],
                "index": 0
            }),
            json!({
                "object": "embedding",
                "embedding": vec![0.0023064255; 1536],
                "index": 2
            }),
            json!({
                "object": "embedding",
                "embedding": vec![0.2; 1536],
                "index": 1
            }),
        ];
        resp_data.append(vec![json!({
            "object": "embedding",
            "embedding": vec![0.2; 1536],
            "index": 1
        }); 256].as_mut());

        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "object": "list",
                "data": resp_data,
                "model": "text-embedding-ada-002",
                "usage": {
                  "prompt_tokens": 8,
                  "total_tokens": 8
                }
            }
        ));
    });
    server.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/embeddings")
            .header("Authorization", "Bearer TEST")
            .header("Content-Type", "application/json")
            .json_body(json!({
                "input": [
                    "My boss says I zindle too much, what does that mean?"
                ],
                "model": "text-embedding-ada-002",
                "dimensions": 1536,
            }));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                    "object": "list",
                    "data": [
                      {
                        "object": "embedding",
                        "embedding": vec![0.0023064254; 1536],
                        "index": 0
                      }
                    ],
                    "model": "text-embedding-ada-002",
                    "usage": {
                      "prompt_tokens": 8,
                      "total_tokens": 8
                    }
                }
            ));
    });

    // Initialize OpenAI client
    let openai_client = openai::Client::builder("TEST")
        .base_url(&server.base_url())
        .build();

    // Select an embedding model.
    let model = openai_client.embedding_model(openai::TEXT_EMBEDDING_ADA_002);

    // Initialize LanceDB locally.
    let db = lancedb::connect("data/lancedb-store")
        .execute()
        .await
        .unwrap();

    // Generate embeddings for the test data.
    let embeddings = EmbeddingsBuilder::new(model.clone())
        .documents(words()).unwrap()
        // Note: need at least 256 rows in order to create an index so copy the definition 256 times for testing purposes.
        .documents(
            (0..256)
                .map(|i| Word {
                    id: format!("doc{i}"),
                    definition: "Definition of *flumbuzzle (noun)*: A sudden, inexplicable urge to rearrange or reorganize small objects, such as desk items or books, for no apparent reason.".to_string()
                })
        ).unwrap()
        .build()
        .await.unwrap();

    let table = db
        .create_table(
            "words",
            RecordBatchIterator::new(
                vec![as_record_batch(embeddings, model.ndims())],
                Arc::new(schema(model.ndims())),
            ),
        )
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

    // Define search_params params that will be used by the vector store to perform the vector search.
    let search_params = SearchParams::default();
    let vector_store_index = LanceDbVectorIndex::new(table, model, "id", search_params)
        .await
        .unwrap();

    let query = "My boss says I zindle too much, what does that mean?";
    let req = VectorSearchRequest::builder()
        .query(query)
        .samples(1)
        .build()
        .expect("VectorSearchRequest should not fail to build here");

    // Query the index
    let results = vector_store_index
        .top_n::<serde_json::Value>(req)
        .await
        .unwrap();

    let (distance, _, value) = &results.first().unwrap();

    assert_eq!(
        *value,
        json!({
            "_distance": distance,
            "definition": "Definition of *zindle (verb)*: to pretend to be working on something important while actually doing something completely unrelated or unproductive.",
            "id": "doc1"
        })
    );

    db.drop_all_tables().await.unwrap();
}
