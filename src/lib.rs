#![cfg_attr(docsrs, feature(doc_cfg))]
//! Public facade for Rig.
//!
//! The `rig` crate is the user-facing entry point for Rig. It re-exports the
//! portable contracts from `rig_core` and the classic runtime from `rig_agent`,
//! preserving the familiar `rig::...` paths.
//!
//! # Companion integrations
//!
//! Companion provider and vector-store crates are exposed as feature-gated
//! modules on this facade. Enable only the integrations your application uses:
//!
//! ```toml
//! [dependencies]
//! rig = { version = "*", features = ["lancedb", "fastembed"] }
//! ```
//!
//! This enables modules such as `rig::lancedb` and `rig::fastembed`. Other
//! companion integrations follow the same pattern, with feature names aligned to
//! their facade module paths wherever Rust module naming allows it.
//!
//! # When to use `rig-core` directly
//!
//! Depend on the `rig-core` package directly when you only need the core Rig
//! implementation crate, including provider abstractions, built-in core
//! providers, tools, memory traits, and vector-store traits, without the root
//! facade's companion integration feature surface.

pub use rig_core::*;

#[cfg(feature = "agent")]
#[cfg_attr(docsrs, doc(cfg(feature = "agent")))]
pub use rig_agent::{Agent, AgentBuilder, AgentRun, AgentRunner, ExtractionResponse};

/// Direct access to the portable provider and data contracts.
pub mod core {
    pub use rig_core::*;
}

/// Classic agent orchestration and lifecycle APIs.
#[cfg(feature = "agent")]
#[cfg_attr(docsrs, doc(cfg(feature = "agent")))]
pub mod agent {
    pub use rig_agent::agent::*;
}

/// Provider clients plus classic agent/extractor constructors.
pub mod client {
    #[cfg(feature = "agent")]
    pub use rig_agent::client::{AgentClientExt, AgentModelExt, CompletionClient};
    pub use rig_core::client::*;
}

/// Low-level completion contracts plus classic prompting traits and errors.
pub mod completion {
    #[cfg(feature = "agent")]
    pub use rig_agent::completion::{
        Chat, Prompt, PromptError, StructuredOutputError, TypedPrompt,
    };
    pub use rig_core::completion::*;
}

/// Classic typed extraction.
#[cfg(feature = "agent")]
#[cfg_attr(docsrs, doc(cfg(feature = "agent")))]
pub mod extractor {
    pub use rig_agent::extractor::*;
}

/// Classic runtime integrations.
#[cfg(feature = "agent")]
#[cfg_attr(docsrs, doc(cfg(feature = "agent")))]
pub mod integrations {
    pub use rig_agent::integrations::*;
}

/// Common classic-runtime imports.
pub mod prelude {
    #[cfg(feature = "derive")]
    pub use crate::Embed;
    #[cfg(feature = "agent")]
    pub use rig_agent::prelude::*;
    #[cfg(not(feature = "agent"))]
    pub use rig_core::prelude::*;
}

/// Low-level streaming values plus classic streaming traits.
pub mod streaming {
    #[cfg(feature = "agent")]
    pub use rig_agent::streaming::{StreamingChat, StreamingPrompt};
    pub use rig_core::streaming::*;
}

/// Contextual classic tools and explicitly named portable tool contracts.
pub mod tool {
    #[cfg(feature = "agent")]
    pub use rig_agent::tool::*;
    #[cfg(not(feature = "agent"))]
    pub use rig_core::tool::*;
    #[cfg(feature = "agent")]
    pub use rig_core::tool::{
        PortableDynamicTool, PortableTool, PortableToolEmbedding, portable_tool_definition,
    };
}

#[cfg(all(feature = "agent", any(test, feature = "test-utils")))]
#[cfg_attr(docsrs, doc(cfg(feature = "test-utils")))]
pub mod test_utils {
    pub use rig_agent::test_utils::*;
}

#[cfg(feature = "derive")]
#[cfg_attr(docsrs, doc(cfg(feature = "derive")))]
pub use rig_derive::rig_tool;
#[cfg(feature = "derive")]
#[cfg_attr(docsrs, doc(cfg(feature = "derive")))]
pub use rig_derive::rig_tool as tool_macro;

/// Conversation memory APIs and optional memory policy helpers.
///
/// This module is always available and re-exports the core memory traits and
/// in-process backend from `rig_core::memory`. Enabling the `memory` feature
/// additionally re-exports policy types from the `rig-memory` companion crate
/// into this same module.
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

/// Local CPU inference with validated Llama/SmolLM2 and native tool-capable Qwen3 models.
#[cfg(feature = "candle")]
#[cfg_attr(docsrs, doc(cfg(feature = "candle")))]
pub mod candle {
    pub use rig_candle::*;
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
