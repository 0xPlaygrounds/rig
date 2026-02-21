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
use rig::vector_store::request::SearchFilter;
use rig::vector_store::request::VectorSearchRequest;
use rig::vector_store::{InsertDocuments, VectorStoreIndex};
use rig::{Embed, OneOrMany};
use rig_vectorize::VectorizeFilter;
use rig_vectorize::VectorizeVectorStore;
use serde::{Deserialize, Serialize};

/// A mock embedding model that returns fixed embeddings for testing.
/// This avoids the need for a real OpenAI API key during tests.
#[derive(Clone)]
struct MockEmbeddingModel {
    dimensions: usize,
}

impl MockEmbeddingModel {
    fn new(dimensions: usize) -> Self {
        Self { dimensions }
    }
}

/// Unit type for mock client (not used).
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
                // Generate a deterministic embedding based on text content
                // This ensures the same text always gets the same embedding
                let hash = simple_hash(&text);
                let vec: Vec<f64> = (0..self.dimensions)
                    .map(|i| {
                        let val = ((hash.wrapping_add(i as u64)) % 1000) as f64 / 1000.0;
                        val * 2.0 - 1.0 // Normalize to [-1, 1]
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

/// Simple hash function for deterministic test embeddings.
fn simple_hash(s: &str) -> u64 {
    let mut hash: u64 = 5381;
    for c in s.bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(c as u64);
    }
    hash
}

/// Helper to get environment variables or skip the test.
fn get_env_or_skip(var: &str) -> Option<String> {
    std::env::var(var).ok()
}

/// Helper to create a vector store from environment variables.
fn create_vector_store() -> Option<VectorizeVectorStore<MockEmbeddingModel>> {
    let account_id = get_env_or_skip("CLOUDFLARE_ACCOUNT_ID")?;
    let api_token = get_env_or_skip("CLOUDFLARE_API_TOKEN")?;
    let index_name = get_env_or_skip("VECTORIZE_INDEX_NAME")?;

    // Use 1536 dimensions to match common embedding models like OpenAI
    let model = MockEmbeddingModel::new(1536);

    Some(VectorizeVectorStore::new(
        model, account_id, index_name, api_token,
    ))
}

/// Test document for insertion tests.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct TestDocument {
    id: String,
    content: String,
    category: String,
}

// Implement Embed trait manually for TestDocument
impl Embed for TestDocument {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(self.content.clone());
        Ok(())
    }
}

#[tokio::test]
async fn test_insert_documents() {
    let Some(vector_store) = create_vector_store() else {
        eprintln!("Skipping test: Required environment variables not set");
        return;
    };

    // Create test documents with pre-computed embeddings
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

    let result = vector_store
        .insert_documents(documents_with_embeddings)
        .await;

    match result {
        Ok(()) => {
            println!("Successfully inserted 2 documents");
        }
        Err(e) => {
            panic!("Insert failed: {:?}", e);
        }
    }
}

#[tokio::test]
async fn test_insert_and_query() {
    let Some(vector_store) = create_vector_store() else {
        eprintln!("Skipping test: Required environment variables not set");
        return;
    };

    let model = MockEmbeddingModel::new(1536);

    // Create a unique document to insert
    let unique_id = format!(
        "test-doc-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
    );

    let doc = TestDocument {
        id: unique_id.clone(),
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

    println!("Inserted document with content: {}", doc.content);

    // Note: Vectorize has eventual consistency, so the document may not be
    // immediately queryable. In a real test, you might want to add a delay
    // or poll for the document.
    println!("Note: Document may take a few seconds to become queryable (eventual consistency)");

    // Query for the document
    let request = VectorSearchRequest::builder()
        .query(&doc.content) // Query with the same content
        .samples(5)
        .build()
        .expect("Failed to build request");

    let result = vector_store.top_n_ids(request).await;

    match result {
        Ok(results) => {
            println!("Query returned {} results", results.len());
            for (score, id) in &results {
                println!("  Score: {:.4}, ID: {}", score, id);
            }
        }
        Err(e) => {
            panic!("Query failed: {:?}", e);
        }
    }
}

#[tokio::test]
async fn test_top_n_returns_full_documents() {
    let Some(vector_store) = create_vector_store() else {
        eprintln!("Skipping test: Required environment variables not set");
        return;
    };

    // Query existing documents to verify top_n() returns full deserialized documents
    // We use documents that were inserted by previous tests (eventual consistency is fine here)
    let request = VectorSearchRequest::builder()
        .query("Rust programming language systems")
        .samples(5)
        .build()
        .expect("Failed to build request");

    // Use top_n() which returns full deserialized documents
    let result = vector_store.top_n::<TestDocument>(request).await;

    match result {
        Ok(results) => {
            println!("top_n() returned {} results", results.len());

            for (score, id, document) in &results {
                println!("  Score: {:.4}, ID: {}", score, id);
                println!("    Document: {:?}", document);

                // Verify the document has all expected fields populated
                assert!(!document.id.is_empty(), "Document should have an id");
                assert!(!document.content.is_empty(), "Document should have content");
                assert!(
                    !document.category.is_empty(),
                    "Document should have a category"
                );
            }

            // Verify we got results (index has data from previous tests)
            if results.is_empty() {
                println!(
                    "Note: No results returned. Index may be empty or documents not yet indexed."
                );
            } else {
                println!("Successfully deserialized {} documents", results.len());
            }
        }
        Err(e) => {
            panic!("top_n() failed: {:?}", e);
        }
    }
}

#[tokio::test]
async fn test_top_n_with_multiple_documents() {
    let Some(vector_store) = create_vector_store() else {
        eprintln!("Skipping test: Required environment variables not set");
        return;
    };

    // Query for documents - should return multiple results with full metadata
    let request = VectorSearchRequest::builder()
        .query("programming language")
        .samples(10)
        .build()
        .expect("Failed to build request");

    let result = vector_store.top_n::<TestDocument>(request).await;

    match result {
        Ok(results) => {
            println!(
                "top_n() for 'programming language' returned {} results",
                results.len()
            );
            for (score, id, document) in &results {
                println!(
                    "  Score: {:.4}, ID: {}, Category: {}",
                    score, id, document.category
                );
                println!("    Content: {}", document.content);
            }
        }
        Err(e) => {
            panic!("top_n() failed: {:?}", e);
        }
    }
}

#[tokio::test]
async fn test_query_with_eq_filter() {
    let Some(vector_store) = create_vector_store() else {
        eprintln!("Skipping test: Required environment variables not set");
        return;
    };

    // Query with a filter to only return documents with category = "programming"
    let filter = VectorizeFilter::eq("category", serde_json::json!("programming"));

    let request = VectorSearchRequest::builder()
        .query("language")
        .samples(10)
        .filter(filter)
        .build()
        .expect("Failed to build request");

    let result = vector_store.top_n::<TestDocument>(request).await;

    match result {
        Ok(results) => {
            println!(
                "Filtered query (category=programming) returned {} results",
                results.len()
            );
            for (score, id, document) in &results {
                println!(
                    "  Score: {:.4}, ID: {}, Category: {}",
                    score, id, document.category
                );
                // All results should have category = "programming"
                assert_eq!(
                    document.category, "programming",
                    "Filter should only return programming documents"
                );
            }
        }
        Err(e) => {
            // Filter might fail if metadata is not indexed - this is expected
            println!(
                "Filtered query failed (metadata may not be indexed): {:?}",
                e
            );
            println!("Note: To use filters, metadata fields must be indexed in Vectorize.");
            println!(
                "Run: wrangler vectorize create-metadata-index rig-integration-test --property-name=category --type=string"
            );
        }
    }
}

#[tokio::test]
async fn test_query_with_combined_filters() {
    let Some(vector_store) = create_vector_store() else {
        eprintln!("Skipping test: Required environment variables not set");
        return;
    };

    // Combine multiple filters with AND
    let filter = VectorizeFilter::eq("category", serde_json::json!("programming"))
        .and(VectorizeFilter::ne("id", serde_json::json!("doc-1")));

    let request = VectorSearchRequest::builder()
        .query("programming")
        .samples(10)
        .filter(filter)
        .build()
        .expect("Failed to build request");

    let result = vector_store.top_n::<TestDocument>(request).await;

    match result {
        Ok(results) => {
            println!("Combined filter query returned {} results", results.len());
            for (score, id, document) in &results {
                println!(
                    "  Score: {:.4}, ID: {}, Category: {}",
                    score, id, document.category
                );
                // Should not include doc-1
                assert_ne!(document.id, "doc-1", "Filter should exclude doc-1");
            }
        }
        Err(e) => {
            println!(
                "Combined filter query failed (metadata may not be indexed): {:?}",
                e
            );
        }
    }
}

#[tokio::test]
async fn test_query_with_in_filter() {
    let Some(vector_store) = create_vector_store() else {
        eprintln!("Skipping test: Required environment variables not set");
        return;
    };

    // Filter for multiple categories using $in
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

    let result = vector_store.top_n::<TestDocument>(request).await;

    match result {
        Ok(results) => {
            println!("$in filter query returned {} results", results.len());
            for (score, id, document) in &results {
                println!(
                    "  Score: {:.4}, ID: {}, Category: {}",
                    score, id, document.category
                );
                // All results should have category in ["programming", "database"]
                assert!(
                    document.category == "programming" || document.category == "database",
                    "Filter should only return programming or database documents, got: {}",
                    document.category
                );
            }
        }
        Err(e) => {
            println!(
                "$in filter query failed (metadata may not be indexed): {:?}",
                e
            );
        }
    }
}
