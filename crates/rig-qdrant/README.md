# Rig-Qdrant
Vector store index integration for [Qdrant](https://qdrant.tech/). This integration supports dense vector retrieval using Rig's embedding providers. It is also extensible to allow all [hybrid queries](https://qdrant.tech/documentation/concepts/hybrid-queries/) supported by Qdrant.

## Installation

```toml
[dependencies]
rig-qdrant = "0.2.5"
rig-core = "0.36.0"
```

The root `rig` facade also exposes this crate behind the `qdrant` feature.

## Examples

See [`examples/qdrant_vector_search.rs`](./examples/qdrant_vector_search.rs)
for an end-to-end example using a Qdrant collection with a Rig embedding model.

Filtered searches use the crate-level `QdrantFilter` type:

```rust
use rig_core::vector_store::request::{SearchFilter, VectorSearchRequest};
use rig_qdrant::QdrantFilter;

let req = VectorSearchRequest::<QdrantFilter>::builder()
    .query("What is a linglingdong?")
    .samples(1)
    .filter(QdrantFilter::eq(
        "id",
        serde_json::json!("f9e17d59-32e5-440c-be02-b2759a654824"),
    ))
    .build();
```
