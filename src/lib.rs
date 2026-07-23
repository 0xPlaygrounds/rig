#![cfg_attr(docsrs, feature(doc_cfg))]
//! Public facade for Rig.
//!
//! The `rig` crate is the user-facing entry point for Rig. It re-exports the
//! portable contracts from `rig_core` at their familiar `rig::...` paths and the
//! classic runtime from `rig_agent` under `rig::agent`.
//!
//! `rig::tool` keeps the classic contextual tool API (`Tool`, `ToolContext`,
//! …) with the default `agent` feature — the same surface as before the runtime
//! split — and always exposes the runtime-independent contracts explicitly as
//! `PortableTool`, `PortableToolEmbedding`, and `PortableDynamicTool`. The
//! classic API also lives at [`crate::agent::tool`]. Classic construction
//! methods such as `client.agent(...)` come from
//! [`crate::client::CompletionClient`], the same import as before the split.
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

    /// Contextual tools for the classic agent runtime.
    pub mod tool {
        pub use rig_agent::tool::*;
    }
}

/// Experimental native-only ECS runtime.
#[cfg(feature = "ecs")]
#[cfg_attr(docsrs, doc(cfg(feature = "ecs")))]
pub mod ecs {
    pub use rig_ecs::*;
}

/// Provider clients plus classic agent/extractor constructors.
pub mod client {
    // The classic runtime's `CompletionClient` extension (adding `agent()` /
    // `extractor()`) owns this path; it shadows the portable provider trait of
    // the same name from the glob below. Provider authors implement the
    // portable trait via `rig_core::client::completion::CompletionClient`.
    #[cfg(feature = "agent")]
    pub use rig_agent::client::{AgentModelExt, CompletionClient};
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

/// Common portable imports plus additive classic-runtime conveniences.
pub mod prelude {
    // The classic contextual `Tool` and its mutable `ToolContext` — the same
    // prelude surface as before the runtime split, so `use rig::prelude::*;
    // impl Tool for X {…}` keeps working.
    #[cfg(feature = "agent")]
    pub use crate::tool::{Tool, ToolContext};
    // The classic `CompletionClient` here intentionally shadows the portable
    // one brought in by the `rig_core::prelude::*` glob below. The classic trait
    // is a superset (it forwards `completion_model` and adds `agent`/
    // `extractor`), so this is the more useful prelude default.
    #[cfg(feature = "agent")]
    pub use rig_agent::prelude::{
        Agent, AgentModelExt, Chat, CompletionClient, MultiTurnStreamItem, Prompt, PromptError,
        StreamingChat, StreamingPrompt, StreamingResult, StructuredOutputError, ToolSet,
        TypedPrompt,
    };
    pub use rig_core::prelude::*;
}

/// Low-level streaming values plus classic streaming traits.
pub mod streaming {
    #[cfg(feature = "agent")]
    pub use rig_agent::streaming::{StreamingChat, StreamingPrompt};
    pub use rig_core::streaming::*;
}

/// Tools for the default (classic) runtime.
///
/// With the `agent` feature (on by default), `Tool`, `ToolContext`, and friends
/// here are the classic *contextual* tool API — the same surface as before the
/// runtime split, so `use rig::tool::{Tool, ToolContext};` keeps working. The
/// runtime-independent portable contracts are always exposed explicitly as
/// [`crate::tool::PortableTool`], [`crate::tool::PortableToolEmbedding`], and
/// [`crate::tool::PortableDynamicTool`] (and in full under
/// [`crate::tool::portable`]). The classic API also lives at
/// [`crate::agent::tool`] for code that prefers the explicit runtime path.
pub mod tool {
    // Canonical execution values — portable, always available.
    pub use rig_core::tool::{
        IntoToolOutput, ToolErrorKind, ToolExecutionError, ToolOutput, ToolResult,
    };
    // Runtime-independent portable contracts — explicit, always available.
    pub use rig_core::tool::{
        PortableDynamicTool, PortableTool, PortableToolEmbedding, portable_tool_definition,
    };
    // Built-in portable tools (e.g. `ThinkTool`), always available.
    pub use rig_core::tool::builtin;

    // Classic contextual tool API (default runtime). `Tool`/`ToolContext` are
    // the classic contextual trait and its mutable context; none of these
    // collide with the portable exports above.
    #[cfg(all(feature = "agent", feature = "rmcp"))]
    #[cfg_attr(docsrs, doc(cfg(feature = "rmcp")))]
    pub use rig_agent::tool::rmcp;
    #[cfg(feature = "agent")]
    #[cfg_attr(docsrs, doc(cfg(feature = "agent")))]
    pub use rig_agent::tool::{
        DynamicTool, MissingToolContext, Tool, ToolContext, ToolEmbedding, ToolSet, ToolSetBuilder,
        server, tool_definition,
    };

    /// The complete portable `rig-core` tool surface, under one explicit path.
    pub mod portable {
        pub use rig_core::tool::*;
    }
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
