//! Public facade for Rig.

pub use rig_core::*;

#[cfg(feature = "bedrock")]
pub mod bedrock {
    pub use rig_bedrock::*;
}

#[cfg(any(
    feature = "fastembed",
    feature = "fastembed-hf-hub",
    feature = "fastembed-hf-hub-native-tls",
    feature = "fastembed-native-tls",
    feature = "fastembed-ort-download-binaries",
))]
pub mod fastembed {
    pub use rig_fastembed::*;
}

#[cfg(feature = "gemini-grpc")]
pub mod gemini_grpc {
    pub use rig_gemini_grpc::*;
}

#[cfg(any(feature = "helixdb", feature = "helixdb-native-tls",))]
pub mod helixdb {
    pub use rig_helixdb::*;
}

#[cfg(any(feature = "lancedb", feature = "lancedb-native-tls",))]
pub mod lancedb {
    pub use rig_lancedb::*;
}

#[cfg(any(feature = "milvus", feature = "milvus-native-tls",))]
pub mod milvus {
    pub use rig_milvus::*;
}

#[cfg(feature = "mongodb")]
pub mod mongodb {
    pub use rig_mongodb::*;
}

#[cfg(feature = "neo4j")]
pub mod neo4j {
    pub use rig_neo4j::*;
}

#[cfg(feature = "postgres")]
pub mod postgres {
    pub use rig_postgres::*;
}

#[cfg(feature = "qdrant")]
pub mod qdrant {
    pub use rig_qdrant::*;
}

#[cfg(feature = "s3vectors")]
pub mod s3vectors {
    pub use rig_s3vectors::*;
}

#[cfg(feature = "scylladb")]
pub mod scylladb {
    pub use rig_scylladb::*;
}

#[cfg(feature = "sqlite")]
pub mod sqlite {
    pub use rig_sqlite::*;
}

#[cfg(feature = "surrealdb")]
pub mod surrealdb {
    pub use rig_surrealdb::*;
}

#[cfg(feature = "vectorize")]
pub mod vectorize {
    pub use rig_vectorize::*;
}

#[cfg(feature = "vertexai")]
pub mod vertexai {
    pub use rig_vertexai::*;
}
