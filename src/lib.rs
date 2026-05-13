#![cfg_attr(docsrs, feature(doc_cfg))]
//! Public facade for Rig.

pub use rig_core::*;

pub mod memory {
    pub use rig_core::memory::*;

    #[cfg(feature = "memory")]
    #[cfg_attr(docsrs, doc(cfg(feature = "memory")))]
    pub use rig_memory::*;
}

#[cfg(feature = "bedrock")]
#[cfg_attr(docsrs, doc(cfg(feature = "bedrock")))]
pub mod bedrock {
    pub use rig_bedrock::*;
}

#[cfg(any(
    feature = "fastembed",
    feature = "fastembed-hf-hub",
    feature = "fastembed-ort-download-binaries",
))]
#[cfg_attr(
    docsrs,
    doc(cfg(any(
        feature = "fastembed",
        feature = "fastembed-hf-hub",
        feature = "fastembed-ort-download-binaries"
    )))
)]
pub mod fastembed {
    pub use rig_fastembed::*;
}

#[cfg(feature = "gemini-grpc")]
#[cfg_attr(docsrs, doc(cfg(feature = "gemini-grpc")))]
pub mod gemini_grpc {
    pub use rig_gemini_grpc::*;
}

#[cfg(feature = "helixdb")]
#[cfg_attr(docsrs, doc(cfg(feature = "helixdb")))]
pub mod helixdb {
    pub use rig_helixdb::*;
}

#[cfg(feature = "lancedb")]
#[cfg_attr(docsrs, doc(cfg(feature = "lancedb")))]
pub mod lancedb {
    pub use rig_lancedb::*;
}

#[cfg(feature = "milvus")]
#[cfg_attr(docsrs, doc(cfg(feature = "milvus")))]
pub mod milvus {
    pub use rig_milvus::*;
}

#[cfg(feature = "mongodb")]
#[cfg_attr(docsrs, doc(cfg(feature = "mongodb")))]
pub mod mongodb {
    pub use rig_mongodb::*;
}

#[cfg(feature = "neo4j")]
#[cfg_attr(docsrs, doc(cfg(feature = "neo4j")))]
pub mod neo4j {
    pub use rig_neo4j::*;
}

#[cfg(feature = "postgres")]
#[cfg_attr(docsrs, doc(cfg(feature = "postgres")))]
pub mod postgres {
    pub use rig_postgres::*;
}

#[cfg(feature = "qdrant")]
#[cfg_attr(docsrs, doc(cfg(feature = "qdrant")))]
pub mod qdrant {
    pub use rig_qdrant::*;
}

#[cfg(feature = "s3vectors")]
#[cfg_attr(docsrs, doc(cfg(feature = "s3vectors")))]
pub mod s3vectors {
    pub use rig_s3vectors::*;
}

#[cfg(feature = "scylladb")]
#[cfg_attr(docsrs, doc(cfg(feature = "scylladb")))]
pub mod scylladb {
    pub use rig_scylladb::*;
}

#[cfg(feature = "sqlite")]
#[cfg_attr(docsrs, doc(cfg(feature = "sqlite")))]
pub mod sqlite {
    pub use rig_sqlite::*;
}

#[cfg(feature = "surrealdb")]
#[cfg_attr(docsrs, doc(cfg(feature = "surrealdb")))]
pub mod surrealdb {
    pub use rig_surrealdb::*;
}

#[cfg(feature = "vectorize")]
#[cfg_attr(docsrs, doc(cfg(feature = "vectorize")))]
pub mod vectorize {
    pub use rig_vectorize::*;
}

#[cfg(feature = "vertexai")]
#[cfg_attr(docsrs, doc(cfg(feature = "vertexai")))]
pub mod vertexai {
    pub use rig_vertexai::*;
}
