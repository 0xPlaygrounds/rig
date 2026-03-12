# Rig-Redis

Vector store index integration for [Redis](https://redis.io/) using RediSearch vector similarity search. This integration supports dense vector retrieval using Rig's embedding providers and leverages Redis's FT.SEARCH command with KNN queries for efficient similarity search.

## Features

- Vector similarity search using Redis's RediSearch module
- Support for KNN (k-nearest neighbors) queries
- Metadata filtering with Redis query syntax
- Document insertion with automatic embedding storage
- Compatible with Redis 7.2+ or Redis Stack

## Prerequisites

You need a Redis instance with RediSearch module enabled. This can be:
- [Redis Stack](https://redis.io/docs/stack/)
- Redis 7.2+ with RediSearch module loaded
- Redis Cloud with RediSearch enabled

## Creating a Vector Index

Before using the vector store, you need to create a RediSearch index with a vector field. Here's an example using redis-cli:

```bash
FT.CREATE word_idx
  ON HASH
  PREFIX 1 doc:
  SCHEMA
    document TEXT
    embedded_text TEXT
    embedding VECTOR FLAT 6
      TYPE FLOAT32
      DIM 1536
      DISTANCE_METRIC COSINE
```

Replace `1536` with your embedding model's dimensionality.

## Usage Example

```rust
use rig::providers::openai;
use rig::vector_store::{InsertDocuments, VectorStoreIndex};
use rig_redis::RedisVectorStore;

// Create embedding model
let openai_client = openai::Client::from_env();
let model = openai_client.embedding_model(openai::TEXT_EMBEDDING_3_SMALL);

// Create Redis client
let redis_client = redis::Client::open("redis://127.0.0.1:6379")?;

// Create vector store
let vector_store = RedisVectorStore::new(
    model,
    redis_client,
    "word_idx".to_string(),      // index name
    "embedding".to_string(),      // vector field name
);

// Insert documents
vector_store.insert_documents(documents).await?;

// Search
let results = vector_store
    .top_n::<MyDocument>(
        VectorSearchRequest::builder()
            .query("your search query")
            .samples(5)
            .build()?
    )
    .await?;
```

You can find complete examples [here](https://github.com/0xPlaygrounds/rig/tree/main/rig-integrations/rig-redis/examples).

## Distance Metrics

Redis supports three distance metrics:
- **COSINE** - Cosine similarity (default, recommended)
- **L2** - Euclidean distance
- **IP** - Inner product

Choose the metric that matches your embedding model when creating the index.

## Limitations

- Requires pre-created RediSearch index
- Vector dimensionality must match the index definition
- Embeddings are stored as FLOAT32 (converted from FLOAT64)

## Testing

### Prerequisites

Integration tests require Docker to be running, as they use testcontainers to spin up a Redis Stack instance.

### Running Tests

```bash
# Run all tests (unit + integration)
cargo test

# Run only unit tests
cargo test --lib

# Run only integration tests
cargo test --test integration_tests

# Or use the Makefile
make test              # All tests
make test-unit         # Unit tests only
make test-integration  # Integration tests only
```

### Manual Testing with Local Redis

You can start a local Redis Stack instance for manual testing:

```bash
# Start Redis Stack
make redis-local
# or
docker run -d --name redis-stack -p 6379:6379 redis/redis-stack:latest

# Create a test index
redis-cli FT.CREATE word_idx ON HASH SCHEMA document TEXT embedded_text TEXT embedding VECTOR FLAT 6 TYPE FLOAT32 DIM 1536 DISTANCE_METRIC COSINE

# Run the example
make run-example
# or
cargo run --example vector_search_redis

# Stop Redis Stack
make redis-stop
```
