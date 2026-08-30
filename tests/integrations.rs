#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used,
    clippy::unreachable
)]

// `bedrock` and `vectorize` are the two suites that do not build an OpenAI
// client; the rest embed one, so they need both features.
#[cfg(feature = "bedrock")]
#[path = "integrations/bedrock/mod.rs"]
mod bedrock;
#[cfg(all(feature = "lancedb", feature = "openai"))]
#[path = "integrations/lancedb/mod.rs"]
mod lancedb;
#[cfg(all(feature = "mongodb", feature = "openai"))]
#[path = "integrations/mongodb.rs"]
mod mongodb;
#[cfg(all(feature = "neo4j", feature = "openai"))]
#[path = "integrations/neo4j.rs"]
mod neo4j;
#[cfg(all(feature = "postgres", feature = "openai"))]
#[path = "integrations/postgres.rs"]
mod postgres;
#[cfg(all(feature = "qdrant", feature = "openai"))]
#[path = "integrations/qdrant.rs"]
mod qdrant;
#[cfg(all(feature = "scylladb", feature = "openai"))]
#[path = "integrations/scylladb.rs"]
mod scylladb;
#[cfg(all(feature = "sqlite", feature = "openai"))]
#[path = "integrations/sqlite.rs"]
mod sqlite;
#[cfg(feature = "vectorize")]
#[path = "integrations/vectorize.rs"]
mod vectorize;
