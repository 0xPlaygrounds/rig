# Rig-Weaviate

Vector store index integration for [Weaviate](https://weaviate.io/). Rig embeds
documents with its own embedding models, inserts them through Weaviate's
[batch import](https://docs.weaviate.io/weaviate/manage-objects/import) REST
endpoint and searches them with a GraphQL
[`nearVector`](https://docs.weaviate.io/weaviate/api/graphql/search-operators#nearvector)
query. Weaviate has no official Rust client, so this crate talks HTTP through
any Rig `HttpClientExt` transport.

## Installation

```toml
[dependencies]
rig-weaviate = "0.44.0"
rig-core = "0.44.0"
```

The root `rig` facade also exposes this crate as `rig::weaviate` behind the
`weaviate` feature.

## Data model

Each embedding becomes one Weaviate object in the store's collection. The
object keeps the whole document as JSON text in a `document` property, and
`top_n` decodes that property. Inserting into a missing collection creates it
when the server's auto-schema is enabled, which is Weaviate's default. A
collection with several named vectors needs `target_vector(name)`, and a
named vector must be declared when the collection is created.

Scores are `1 - distance`. Under Weaviate's default cosine metric that is the
cosine similarity, so a request threshold is a minimum similarity.

## Filters

`WeaviateFilter` builds a GraphQL `where` filter over the collection's
properties. Property names and values are escaped. Weaviate checks each value
against the property's data type, so use `WeaviateValue::Int` for `int`
properties and `WeaviateValue::Number` for `number` properties.

```rust
use rig_core::vector_store::request::{SearchFilter, VectorSearchRequest};
use rig_weaviate::WeaviateFilter;

let req = VectorSearchRequest::<WeaviateFilter>::builder()
    .query("What is a linglingdong?")
    .samples(2)
    .filter(WeaviateFilter::like("document", "*moon*"))
    .build();
```

## Example

[`examples/vector_search_weaviate.rs`](./examples/vector_search_weaviate.rs)
inserts and searches three definitions against a local Weaviate:

```bash
docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.40.1
OPENAI_API_KEY=... cargo run -p rig-weaviate --example vector_search_weaviate
```

Set `WEAVIATE_URL` and `WEAVIATE_API_KEY` to use another instance, such as a
Weaviate Cloud cluster.
