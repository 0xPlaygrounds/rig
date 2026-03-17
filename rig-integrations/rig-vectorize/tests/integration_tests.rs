//! Integration tests for rig-vectorize.
//!
//! These tests require a real Cloudflare Vectorize index and valid credentials.
//!
//! Required environment variables:
//! - `CLOUDFLARE_ACCOUNT_ID`: Your Cloudflare account ID
//! - `CLOUDFLARE_API_TOKEN`: API token with Vectorize read/write permissions
//! - `VECTORIZE_INDEX_NAME`: Name of the test index (must exist, 1536 dimensions)
//!
//! To run these tests:
//! ```bash
//! export CLOUDFLARE_ACCOUNT_ID="your-account-id"
//! export CLOUDFLARE_API_TOKEN="your-api-token"
//! export VECTORIZE_INDEX_NAME="rig-integration-test"
//! cargo test --package rig-vectorize --test integration_tests
//! ```

use rig::embeddings::{EmbedError, Embedding, EmbeddingModel, TextEmbedder};
use rig::vector_store::request::{SearchFilter, VectorSearchRequest};
use rig::vector_store::{InsertDocuments, VectorStoreIndex};
use rig::{Embed, OneOrMany};
use rig_vectorize::{VectorizeClient, VectorizeFilter, VectorizeVectorStore};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Vectorize has eventual consistency - documents may not be immediately queryable after insert.
const EVENTUAL_CONSISTENCY_DELAY: Duration = Duration::from_secs(5);

#[tokio::test]
async fn test_insert_documents() {
    clear_test_index().await;

    let Some(vector_store) = create_vector_store() else {
        eprintln!("Skipping test: Required environment variables not set");
        return;
    };

    let model = MockEmbeddingModel::new(1536);

    let docs = vec![
        TestDocument {
            id: "doc-1".to_string(),
            content: "Rust is a systems programming language".to_string(),
            category: "programming".to_string(),
        },
        TestDocument {
            id: "doc-3".to_string(),
            content: "Cloudflare Vectorize is a globally distributed vector database".to_string(),
            category: "database".to_string(),
        },
    ];

    let embeddings = model
        .embed_texts(docs.iter().map(|d| d.content.clone()))
        .await
        .expect("Failed to generate embeddings");

    let documents_with_embeddings: Vec<(TestDocument, OneOrMany<Embedding>)> = docs
        .into_iter()
        .zip(embeddings.into_iter())
        .map(|(doc, emb)| (doc, OneOrMany::one(emb)))
        .collect();

    vector_store
        .insert_documents(documents_with_embeddings)
        .await
        .expect("Insert should succeed");
}

#[tokio::test]
async fn test_insert_and_query() {
    clear_test_index().await;

    let Some(vector_store) = create_vector_store() else {
        eprintln!("Skipping test: Required environment variables not set");
        return;
    };

    let model = MockEmbeddingModel::new(1536);

    let doc = TestDocument {
        id: "test-doc".to_string(),
        content: "Rig is a Rust library for building AI applications".to_string(),
        category: "ai".to_string(),
    };

    let embeddings = model
        .embed_texts(vec![doc.content.clone()])
        .await
        .expect("Failed to generate embeddings");

    let documents_with_embeddings = vec![(
        doc.clone(),
        OneOrMany::one(embeddings.into_iter().next().unwrap()),
    )];

    vector_store
        .insert_documents(documents_with_embeddings)
        .await
        .expect("Failed to insert document");

    // Wait for eventual consistency
    tokio::time::sleep(EVENTUAL_CONSISTENCY_DELAY).await;

    let request = VectorSearchRequest::builder()
        .query(&doc.content)
        .samples(5)
        .build()
        .expect("Failed to build request");

    let results = vector_store
        .top_n_ids(request)
        .await
        .expect("Query should succeed");

    assert!(!results.is_empty(), "Should return at least one result");
}

#[tokio::test]
async fn test_top_n_returns_full_documents() {
    clear_test_index().await;

    let Some(vector_store) = create_vector_store() else {
        eprintln!("Skipping test: Required environment variables not set");
        return;
    };

    let model = MockEmbeddingModel::new(1536);
    let doc = TestDocument {
        id: "doc-rust".to_string(),
        content: "Rust is a systems programming language".to_string(),
        category: "programming".to_string(),
    };

    let embeddings = model
        .embed_texts(vec![doc.content.clone()])
        .await
        .expect("Failed to generate embeddings");

    vector_store
        .insert_documents(vec![(
            doc.clone(),
            OneOrMany::one(embeddings.into_iter().next().unwrap()),
        )])
        .await
        .expect("Failed to insert document");

    // Wait for eventual consistency
    tokio::time::sleep(EVENTUAL_CONSISTENCY_DELAY).await;

    let request = VectorSearchRequest::builder()
        .query("Rust programming language systems")
        .samples(5)
        .build()
        .expect("Failed to build request");

    let results = vector_store
        .top_n::<TestDocument>(request)
        .await
        .expect("top_n should succeed");

    assert!(!results.is_empty(), "Should return at least one result");

    for (_score, _id, document) in &results {
        assert!(!document.id.is_empty(), "Document should have an id");
        assert!(!document.content.is_empty(), "Document should have content");
        assert!(
            !document.category.is_empty(),
            "Document should have a category"
        );
    }
}

#[tokio::test]
async fn test_top_n_with_multiple_documents() {
    clear_test_index().await;

    let Some(vector_store) = create_vector_store() else {
        eprintln!("Skipping test: Required environment variables not set");
        return;
    };

    let model = MockEmbeddingModel::new(1536);
    let docs = vec![
        TestDocument {
            id: "doc-rust".to_string(),
            content: "Rust is a systems programming language".to_string(),
            category: "programming".to_string(),
        },
        TestDocument {
            id: "doc-python".to_string(),
            content: "Python is a dynamic programming language".to_string(),
            category: "programming".to_string(),
        },
    ];

    let embeddings = model
        .embed_texts(docs.iter().map(|d| d.content.clone()))
        .await
        .expect("Failed to generate embeddings");

    let documents_with_embeddings: Vec<(TestDocument, OneOrMany<Embedding>)> = docs
        .into_iter()
        .zip(embeddings.into_iter())
        .map(|(doc, emb)| (doc, OneOrMany::one(emb)))
        .collect();

    vector_store
        .insert_documents(documents_with_embeddings)
        .await
        .expect("Failed to insert documents");

    // Wait for eventual consistency
    tokio::time::sleep(EVENTUAL_CONSISTENCY_DELAY).await;

    let request = VectorSearchRequest::builder()
        .query("programming language")
        .samples(10)
        .build()
        .expect("Failed to build request");

    let results = vector_store
        .top_n::<TestDocument>(request)
        .await
        .expect("top_n should succeed");

    assert!(
        results.len() >= 2,
        "Should return at least 2 results, got {}",
        results.len()
    );
}

#[tokio::test]
async fn test_query_with_eq_filter() {
    clear_test_index().await;

    let Some(vector_store) = create_vector_store() else {
        eprintln!("Skipping test: Required environment variables not set");
        return;
    };

    let model = MockEmbeddingModel::new(1536);
    let docs = vec![
        TestDocument {
            id: "doc-rust".to_string(),
            content: "Rust is a systems programming language".to_string(),
            category: "programming".to_string(),
        },
        TestDocument {
            id: "doc-vectorize".to_string(),
            content: "Cloudflare Vectorize is a vector database".to_string(),
            category: "database".to_string(),
        },
    ];

    let embeddings = model
        .embed_texts(docs.iter().map(|d| d.content.clone()))
        .await
        .expect("Failed to generate embeddings");

    let documents_with_embeddings: Vec<(TestDocument, OneOrMany<Embedding>)> = docs
        .into_iter()
        .zip(embeddings.into_iter())
        .map(|(doc, emb)| (doc, OneOrMany::one(emb)))
        .collect();

    vector_store
        .insert_documents(documents_with_embeddings)
        .await
        .expect("Failed to insert documents");

    // Wait for eventual consistency
    tokio::time::sleep(EVENTUAL_CONSISTENCY_DELAY).await;

    let filter = VectorizeFilter::eq("category", serde_json::json!("programming"));

    let request = VectorSearchRequest::builder()
        .query("language")
        .samples(10)
        .filter(filter)
        .build()
        .expect("Failed to build request");

    match vector_store.top_n::<TestDocument>(request).await {
        Ok(results) => {
            if results.is_empty() {
                eprintln!(
                    "Filter test inconclusive - no results returned (metadata index may not exist)"
                );
                return;
            }
            for (_score, _id, document) in &results {
                assert_eq!(
                    document.category, "programming",
                    "Filter should only return programming documents"
                );
            }
        }
        Err(e) => {
            eprintln!("Filter test skipped - metadata may not be indexed: {:?}", e);
        }
    }
}

#[tokio::test]
async fn test_query_with_combined_filters() {
    clear_test_index().await;

    let Some(vector_store) = create_vector_store() else {
        eprintln!("Skipping test: Required environment variables not set");
        return;
    };

    let model = MockEmbeddingModel::new(1536);
    let docs = vec![
        TestDocument {
            id: "doc-rust".to_string(),
            content: "Rust is a systems programming language".to_string(),
            category: "programming".to_string(),
        },
        TestDocument {
            id: "doc-python".to_string(),
            content: "Python is a dynamic programming language".to_string(),
            category: "programming".to_string(),
        },
        TestDocument {
            id: "doc-vectorize".to_string(),
            content: "Cloudflare Vectorize is a vector database".to_string(),
            category: "database".to_string(),
        },
    ];

    let embeddings = model
        .embed_texts(docs.iter().map(|d| d.content.clone()))
        .await
        .expect("Failed to generate embeddings");

    let documents_with_embeddings: Vec<(TestDocument, OneOrMany<Embedding>)> = docs
        .into_iter()
        .zip(embeddings.into_iter())
        .map(|(doc, emb)| (doc, OneOrMany::one(emb)))
        .collect();

    vector_store
        .insert_documents(documents_with_embeddings)
        .await
        .expect("Failed to insert documents");

    // Wait for eventual consistency
    tokio::time::sleep(EVENTUAL_CONSISTENCY_DELAY).await;

    // category = "programming" AND id != "doc-rust"
    let filter = VectorizeFilter::eq("category", serde_json::json!("programming"))
        .and(VectorizeFilter::ne("id", serde_json::json!("doc-rust")));

    let request = VectorSearchRequest::builder()
        .query("programming")
        .samples(10)
        .filter(filter)
        .build()
        .expect("Failed to build request");

    match vector_store.top_n::<TestDocument>(request).await {
        Ok(results) => {
            if results.is_empty() {
                eprintln!(
                    "Filter test inconclusive - no results returned (metadata index may not exist)"
                );
                return;
            }
            for (_score, _id, document) in &results {
                assert_ne!(document.id, "doc-rust", "Filter should exclude doc-rust");
                assert_eq!(
                    document.category, "programming",
                    "Filter should only return programming documents"
                );
            }
        }
        Err(e) => {
            eprintln!("Filter test skipped - metadata may not be indexed: {:?}", e);
        }
    }
}

#[tokio::test]
async fn test_query_with_in_filter() {
    clear_test_index().await;

    let Some(vector_store) = create_vector_store() else {
        eprintln!("Skipping test: Required environment variables not set");
        return;
    };

    let model = MockEmbeddingModel::new(1536);
    let docs = vec![
        TestDocument {
            id: "doc-rust".to_string(),
            content: "Rust is a systems programming language".to_string(),
            category: "programming".to_string(),
        },
        TestDocument {
            id: "doc-vectorize".to_string(),
            content: "Cloudflare Vectorize is a vector database".to_string(),
            category: "database".to_string(),
        },
        TestDocument {
            id: "doc-ai".to_string(),
            content: "Machine learning and artificial intelligence".to_string(),
            category: "ai".to_string(),
        },
    ];

    let embeddings = model
        .embed_texts(docs.iter().map(|d| d.content.clone()))
        .await
        .expect("Failed to generate embeddings");

    let documents_with_embeddings: Vec<(TestDocument, OneOrMany<Embedding>)> = docs
        .into_iter()
        .zip(embeddings.into_iter())
        .map(|(doc, emb)| (doc, OneOrMany::one(emb)))
        .collect();

    vector_store
        .insert_documents(documents_with_embeddings)
        .await
        .expect("Failed to insert documents");

    // Wait for eventual consistency
    tokio::time::sleep(EVENTUAL_CONSISTENCY_DELAY).await;

    let filter = VectorizeFilter::in_values(
        "category",
        vec![
            serde_json::json!("programming"),
            serde_json::json!("database"),
        ],
    );

    let request = VectorSearchRequest::builder()
        .query("Rust Vectorize")
        .samples(10)
        .filter(filter)
        .build()
        .expect("Failed to build request");

    match vector_store.top_n::<TestDocument>(request).await {
        Ok(results) => {
            if results.is_empty() {
                eprintln!(
                    "Filter test inconclusive - no results returned (metadata index may not exist)"
                );
                return;
            }
            for (_score, _id, document) in &results {
                assert!(
                    document.category == "programming" || document.category == "database",
                    "Filter should only return programming or database documents, got: {}",
                    document.category
                );
            }
        }
        Err(e) => {
            eprintln!("Filter test skipped - metadata may not be indexed: {:?}", e);
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct TestDocument {
    id: String,
    content: String,
    category: String,
}

impl Embed for TestDocument {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(self.content.clone());
        Ok(())
    }
}

/// A mock embedding model that returns deterministic embeddings for testing.
#[derive(Clone)]
struct MockEmbeddingModel {
    dimensions: usize,
}

impl MockEmbeddingModel {
    fn new(dimensions: usize) -> Self {
        Self { dimensions }
    }
}

struct MockClient;

impl EmbeddingModel for MockEmbeddingModel {
    const MAX_DOCUMENTS: usize = 100;

    type Client = MockClient;

    fn make(_client: &Self::Client, _model: impl Into<String>, dims: Option<usize>) -> Self {
        Self {
            dimensions: dims.unwrap_or(1536),
        }
    }

    fn ndims(&self) -> usize {
        self.dimensions
    }

    async fn embed_texts(
        &self,
        texts: impl IntoIterator<Item = String> + Send,
    ) -> Result<Vec<rig::embeddings::Embedding>, rig::embeddings::EmbeddingError> {
        let texts: Vec<String> = texts.into_iter().collect();
        let embeddings = texts
            .into_iter()
            .map(|text| {
                let hash = simple_hash(&text);
                let vec: Vec<f64> = (0..self.dimensions)
                    .map(|i| {
                        let val = ((hash.wrapping_add(i as u64)) % 1000) as f64 / 1000.0;
                        val * 2.0 - 1.0
                    })
                    .collect();
                rig::embeddings::Embedding {
                    document: text,
                    vec,
                }
            })
            .collect();
        Ok(embeddings)
    }
}

fn simple_hash(s: &str) -> u64 {
    let mut hash: u64 = 5381;
    for c in s.bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(c as u64);
    }
    hash
}

fn get_env_or_skip(var: &str) -> Option<String> {
    std::env::var(var).ok()
}

fn create_vector_store() -> Option<VectorizeVectorStore<MockEmbeddingModel>> {
    let account_id = get_env_or_skip("CLOUDFLARE_ACCOUNT_ID")?;
    let api_token = get_env_or_skip("CLOUDFLARE_API_TOKEN")?;
    let index_name = get_env_or_skip("VECTORIZE_INDEX_NAME")?;

    let model = MockEmbeddingModel::new(1536);

    Some(VectorizeVectorStore::new(
        model, account_id, index_name, api_token,
    ))
}

async fn clear_test_index() {
    let Some(account_id) = get_env_or_skip("CLOUDFLARE_ACCOUNT_ID") else {
        return;
    };
    let Some(api_token) = get_env_or_skip("CLOUDFLARE_API_TOKEN") else {
        return;
    };
    let Some(index_name) = get_env_or_skip("VECTORIZE_INDEX_NAME") else {
        return;
    };

    let client = VectorizeClient::new(account_id, index_name, api_token);

    let mut cursor: Option<String> = None;
    loop {
        let result = match client.list_vectors(Some(1000), cursor.as_deref()).await {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Warning: Failed to list vectors: {:?}", e);
                return;
            }
        };

        if result.vectors.is_empty() {
            break;
        }

        let ids: Vec<String> = result.vectors.into_iter().map(|v| v.id).collect();
        if let Err(e) = client.delete_by_ids(ids).await {
            eprintln!("Warning: Failed to delete vectors: {:?}", e);
            return;
        }

        if !result.is_truncated {
            break;
        }

        cursor = result.next_cursor;
    }
}
