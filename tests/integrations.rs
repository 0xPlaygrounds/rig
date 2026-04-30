#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used,
    clippy::unreachable
)]

#[cfg(feature = "lancedb")]
#[path = "integrations/lancedb/mod.rs"]
mod lancedb;
#[cfg(feature = "mongodb")]
#[path = "integrations/mongodb.rs"]
mod mongodb;
#[cfg(feature = "neo4j")]
#[path = "integrations/neo4j.rs"]
mod neo4j;
#[cfg(feature = "postgres")]
#[path = "integrations/postgres.rs"]
mod postgres;
#[cfg(feature = "qdrant")]
#[path = "integrations/qdrant.rs"]
mod qdrant;
#[cfg(feature = "scylladb")]
#[path = "integrations/scylladb.rs"]
mod scylladb;
#[cfg(feature = "sqlite")]
#[path = "integrations/sqlite.rs"]
mod sqlite;
#[cfg(feature = "vectorize")]
#[path = "integrations/vectorize.rs"]
mod vectorize;
