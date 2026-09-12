#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used,
    clippy::unreachable
)]

#[cfg(feature = "lancedb")]
#[path = "../../tests/integrations/lancedb/mod.rs"]
mod lancedb;
#[cfg(feature = "mongodb")]
#[path = "../../tests/integrations/mongodb.rs"]
mod mongodb;
#[cfg(feature = "neo4j")]
#[path = "../../tests/integrations/neo4j.rs"]
mod neo4j;
#[cfg(feature = "postgres")]
#[path = "../../tests/integrations/postgres.rs"]
mod postgres;
#[cfg(feature = "qdrant")]
#[path = "../../tests/integrations/qdrant.rs"]
mod qdrant;
#[cfg(feature = "scylladb")]
#[path = "../../tests/integrations/scylladb.rs"]
mod scylladb;
#[cfg(feature = "sqlite")]
#[path = "../../tests/integrations/sqlite.rs"]
mod sqlite;
#[cfg(feature = "vectorize")]
#[path = "../../tests/integrations/vectorize.rs"]
mod vectorize;
