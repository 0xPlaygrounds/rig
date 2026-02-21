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

use rig::embeddings::EmbeddingModel;
use rig::vector_store::VectorStoreIndex;
use rig::vector_store::request::VectorSearchRequest;
use rig_vectorize::VectorizeVectorStore;

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
        model,
        account_id,
        index_name,
        api_token,
    ))
}

#[tokio::test]
async fn test_top_n_ids_basic() {
    let Some(vector_store) = create_vector_store() else {
        eprintln!("Skipping test: Required environment variables not set");
        eprintln!("Set CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN, VECTORIZE_INDEX_NAME");
        return;
    };

    // Query for similar vectors
    let request = VectorSearchRequest::builder()
        .query("test query")
        .samples(5)
        .build()
        .expect("Failed to build request");

    let result = vector_store.top_n_ids(request).await;

    println!("Result: {:?}", result);
    match result {
        Ok(results) => {
            println!("Query returned {} results", results.len());
            for (score, id) in &results {
                println!("  Score: {:.4}, ID: {}", score, id);
            }
            // The index might be empty, so we just verify the call succeeded
            assert!(results.len() <= 5, "Should return at most 5 results");
        }
        Err(e) => {
            panic!("Query failed: {:?}", e);
        }
    }
}

#[tokio::test]
async fn test_top_n_ids_with_threshold() {
    let Some(vector_store) = create_vector_store() else {
        eprintln!("Skipping test: Required environment variables not set");
        return;
    };

    // Query with a high threshold - should filter out low-scoring results
    let request = VectorSearchRequest::builder()
        .query("test query with threshold")
        .samples(10)
        .threshold(0.9) // High threshold
        .build()
        .expect("Failed to build request");

    let result = vector_store.top_n_ids(request).await;

    match result {
        Ok(results) => {
            println!("Query with threshold returned {} results", results.len());
            // Verify all results meet the threshold
            for (score, id) in &results {
                println!("  Score: {:.4}, ID: {}", score, id);
                assert!(
                    *score >= 0.9,
                    "Score {} should be >= 0.9 threshold",
                    score
                );
            }
        }
        Err(e) => {
            panic!("Query with threshold failed: {:?}", e);
        }
    }
}
