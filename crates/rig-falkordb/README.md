# rig-falkordb

A FalkorDB vector store for Rig using the official
[`falkordb-rs`](https://github.com/FalkorDB/falkordb-rs) driver and its async API.

## Usage

Use the companion crate directly:

```toml
[dependencies]
rig-falkordb = "0.46.0"
rig-core = "0.46.0"
```

Or enable `falkordb` on the `rig` facade to access the same API at `rig::falkordb`.

```rust,no_run
use rig_core::providers::openai::{self, OpenAI};
use rig_core::vector_store::{VectorStoreIndex, request::VectorSearchRequest};
use rig_falkordb::{FalkorDBClient, vector_index::IndexConfig};

# async fn example() -> anyhow::Result<()> {
let model = OpenAI::from_env()?.embedding(openai::TEXT_EMBEDDING_3_SMALL, None);
let client = FalkorDBClient::connect("falkor://localhost:6379", "documents").await?;
client
    .create_vector_index(IndexConfig::new("Document"), model.capabilities().ndims)
    .await?;
let index = client.get_index(model, "Document", "embedding").await?;
let request = VectorSearchRequest::builder()
    .query("graph databases")
    .samples(3)
    .build();
let results = index.top_n::<serde_json::Value>(request).await?;
# Ok(())
# }
```

FalkorDB indexes are identified by node label and property, not a separate index
name. `create_vector_index` waits for readiness and propagates errors.
`get_index` reads the existing index's dimensions and metric and rejects a
known model dimension mismatch. Always reuse the model that generated the stored
embeddings. Existing indexed properties must contain `vecf32` vectors.

`FalkorDBClient::new` accepts an existing `falkordb::AsyncGraph`, allowing custom
driver settings, TLS, and connection pooling. Enable the driver's
`tokio-rustls` or `tokio-native-tls` feature in your application for TLS.
`connect` accepts authentication in the connection URI.

## Inserting and searching

`InsertDocuments::insert_documents` accepts the output of Rig's
`EmbeddingsBuilder`. It creates one node per embedding, stores object fields as
node properties, and adds `embedded_text` and the configured embedding property.
Non-object documents are stored under `document`. These two embedding-related
property names are reserved. Fields must be supported FalkorDB property values;
nested objects cannot be stored as node properties.

`top_n` returns node properties with the embedding property set to null.
`top_n_ids` returns only scores and node IDs. Both use FalkorDB node IDs, not
the document's `id` property.

Scores are higher-is-better: cosine uses `1 - distance` and Euclidean uses
`1 / (1 + distance)`, where `distance` is the value returned by FalkorDB.
Thresholds are inclusive minimum similarity scores. Filters and thresholds are
applied after the requested nearest-neighbor candidates are selected, so a
filtered search can return fewer than `samples` results.

Compose metadata predicates using `FalkorDBSearchFilter` and Rig's `SearchFilter`
trait (`eq`, `gt`, `lt`, `and`, `or`). Values are driver-bound parameters, names
are quoted, and equality to JSON null matches absent properties.

## Example

Start a local server, set `OPENAI_API_KEY` and `FALKORDB_URI`, then run:

```sh
docker run --rm -p 6379:6379 falkordb/falkordb:v4.22.3
export FALKORDB_URI=falkor://localhost:6379
cargo run -p rig-falkordb --example vector_search_simple
```

The [example](examples/vector_search_simple.rs) generates embeddings, inserts
documents, and runs scored-document and ID-only searches.

The Docker-backed suite uses mocked embeddings and needs no OpenAI credentials:

```sh
cargo test -p rig-service-tests --features falkordb --test integrations falkordb
```
