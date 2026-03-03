# Rig RAG Reference

## Overview

RAG in Rig works by:
1. Embedding documents into a vector store
2. Creating an index from the store
3. Attaching the index to an agent via `.dynamic_context()`

## Step 1: Define Documents

Documents must implement `Embed` (via derive macro) and `Serialize`:

```rust
use rig::Embed;
use serde::{Deserialize, Serialize};

#[derive(Embed, Serialize, Deserialize, Clone, Debug)]
struct Document {
    id: String,

    #[embed]  // This field will be embedded
    content: String,

    metadata: String,  // Non-embedded fields are preserved
}
```

## Step 2: Create Embeddings

```rust
use rig::embeddings::EmbeddingsBuilder;
use rig::providers::openai;

let client = openai::Client::from_env();
let embedding_model = client.embedding_model(openai::TEXT_EMBEDDING_ADA_002);

let documents = vec![
    Document {
        id: "1".into(),
        content: "Rig is a Rust AI framework.".into(),
        metadata: "about".into(),
    },
    Document {
        id: "2".into(),
        content: "Rig supports 19+ LLM providers.".into(),
        metadata: "features".into(),
    },
];

let embeddings = EmbeddingsBuilder::new(embedding_model.clone())
    .documents(documents)?
    .build()
    .await?;
```

## Step 3: Create Vector Store and Index

### In-Memory (built into rig-core)

```rust
use rig::vector_store::in_memory_store::InMemoryVectorStore;

let store = InMemoryVectorStore::from_documents(embeddings);
let index = store.index(embedding_model);
```

### MongoDB

```rust
// Cargo.toml: rig-mongodb = "..."
use rig_mongodb::{MongoDbVectorStore, SearchParams};

let collection = mongodb_client
    .database("my_db")
    .collection("documents");

// Insert embeddings
let store = MongoDbVectorStore::new(collection);
store.add_documents(embeddings).await?;

// Create index
let index = store.index(
    embedding_model,
    "vector_index",  // Atlas Search index name
    SearchParams::new(),
);
```

### LanceDB

```rust
// Cargo.toml: rig-lancedb = "..."
use rig_lancedb::{LanceDbVectorStore, SearchParams};

let db = lancedb::connect("data/lancedb").execute().await?;
let table = db.create_table("documents", embeddings).execute().await?;

let index = LanceDbVectorStore::new(table, embedding_model, SearchParams::default());
```

## Step 4: Attach to Agent

```rust
let rag_agent = client
    .agent(openai::GPT_4O)
    .preamble("Answer questions using the provided context.")
    .dynamic_context(5, index)  // Retrieve top-5 similar docs per query
    .build();

let response = rag_agent.prompt("What is Rig?").await?;
```

## VectorStoreIndex Trait

For implementing custom backends:

```rust
pub trait VectorStoreIndex: WasmCompatSend + WasmCompatSync {
    type Filter: SearchFilter + WasmCompatSend + WasmCompatSync;

    async fn top_n<T: for<'a> Deserialize<'a> + WasmCompatSend>(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String, T)>, VectorStoreError>;

    async fn top_n_ids(
        &self,
        req: VectorSearchRequest<Self::Filter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError>;
}
```

## VectorSearchRequest

Constructed via builder:

```rust
use rig::vector_store::VectorSearchRequest;

let req = VectorSearchRequest::builder()
    .query("search text")
    .samples(5)
    .build()?;
```
