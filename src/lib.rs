#![cfg_attr(docsrs, feature(doc_cfg))]
//! Public facade for Rig.
//!
//! The `rig` crate is the user-facing entry point for Rig. It re-exports the
//! portable contracts from `rig_core` at their familiar `rig::...` paths and the
//! classic runtime from `rig_agent` under `rig::agent`.
//!
//! `rig::tool` exposes the portable, context-free tool contracts —
//! `PortableTool`, `PortableToolEmbedding`, and `PortableDynamicTool` — and
//! aliases `Tool` to `PortableTool` so classic `impl Tool for X` sites keep
//! compiling once their `call` signature drops the removed `ToolContext`
//! parameter. The same surface also lives at [`crate::agent::tool`]. Classic
//! construction methods such as `client.agent(...)` come from
//! [`crate::client::AgentClientExt`]; `use rig::prelude::*;` brings it in
//! alongside the canonical `CompletionClient`, the same surface as before the
//! split.
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
//! providers, tools, memory traits, and vector-store types, without the root
//! facade's companion integration feature surface.

pub use rig_core::*;

#[cfg(feature = "agent")]
#[cfg_attr(docsrs, doc(cfg(feature = "agent")))]
pub use rig_agent::{Agent, AgentBuilder, AgentRun, AgentRunner};

/// Direct access to the portable provider and data contracts.
pub mod core {
    pub use rig_core::*;
}

/// Classic agent orchestration and lifecycle APIs.
#[cfg(feature = "agent")]
#[cfg_attr(docsrs, doc(cfg(feature = "agent")))]
pub mod agent {
    pub use rig_agent::agent::*;

    /// Tool records executed by the classic agent runtime.
    pub mod tool {
        pub use rig_agent::tool::*;
    }
}

/// Provider clients plus the classic agent constructor.
pub mod client {
    // Classic-runtime construction extension: `agent()` on any completion
    // client (`AgentClientExt`), plus `ToProviderConfig` for capturing a
    // client's connection details as plain configuration. Extraction is the
    // free-function surface in [`crate::extract`].
    #[cfg(feature = "agent")]
    pub use rig_agent::client::{AgentClientExt, ToProviderConfig};

    // The full portable provider-client surface, including the canonical
    // `CompletionClient`. `AgentClientExt` is a distinct name, so there is no
    // shadow — just one canonical completion-client trait plus the classic
    // construction extension.
    pub use rig_core::client::*;
}

/// Low-level completion contracts plus the classic runtime's errors.
pub mod completion {
    #[cfg(feature = "agent")]
    pub use rig_agent::completion::{PromptError, StructuredOutputError};
    pub use rig_core::completion::*;
}

/// Classic runtime integrations.
#[cfg(feature = "agent")]
#[cfg_attr(docsrs, doc(cfg(feature = "agent")))]
pub mod integrations {
    pub use rig_agent::integrations::*;
}

/// The bundled provider set as plain configuration, plus the live-handle
/// runtime — the data-oriented fulfilment layer.
#[cfg(feature = "agent")]
#[cfg_attr(docsrs, doc(cfg(feature = "agent")))]
pub use rig_agent::provider;

/// The blocking session driver over the sans-IO run protocol.
#[cfg(feature = "agent")]
#[cfg_attr(docsrs, doc(cfg(feature = "agent")))]
pub use rig_agent::session;

/// The streaming session driver over the sans-IO run protocol.
#[cfg(feature = "agent")]
#[cfg_attr(docsrs, doc(cfg(feature = "agent")))]
pub use rig_agent::stream;

/// Automatic tool execution over the session drivers.
#[cfg(feature = "agent")]
#[cfg_attr(docsrs, doc(cfg(feature = "agent")))]
pub use rig_agent::executor;

/// Structured extraction over the session runtime.
#[cfg(feature = "agent")]
#[cfg_attr(docsrs, doc(cfg(feature = "agent")))]
pub use rig_agent::extract;

/// Concrete, attach-and-forget hooks for the session drivers.
#[cfg(feature = "agent")]
#[cfg_attr(docsrs, doc(cfg(feature = "agent")))]
pub use rig_agent::hooks;

/// The thin, forward-looking agent over the session drivers.
#[cfg(feature = "agent")]
#[cfg_attr(docsrs, doc(cfg(feature = "agent")))]
pub use rig_agent::agent_api;

/// The forward-looking concrete agent type (renamed to `Agent` once the
/// classic runtime is retired).
#[cfg(feature = "agent")]
#[cfg_attr(docsrs, doc(cfg(feature = "agent")))]
pub use rig_agent::agent_api::SessionAgent;

/// Common portable imports plus additive classic-runtime conveniences.
pub mod prelude {
    // `Tool` is an alias for the portable, context-free `PortableTool`, so
    // `use rig::prelude::*; impl Tool for X {…}` keeps working once the
    // implementation's `call` drops the removed `ToolContext` parameter.
    pub use crate::tool::Tool;
    // The classic construction extension `AgentClientExt` (adding `agent()`)
    // sits alongside the canonical `CompletionClient` brought in
    // by the `rig_core::prelude::*` glob below. The two traits share no method
    // names, so both resolve without ambiguity and together restore the
    // pre-split `client.completion_model(m)` / `client.agent(m)` surface.
    #[cfg(feature = "agent")]
    pub use rig_agent::prelude::{
        Agent, AgentClientExt, MultiTurnStreamItem, PromptError, StreamingResult,
        StructuredOutputError, ToProviderConfig,
    };
    pub use rig_core::prelude::*;
}

/// Low-level streaming values.
pub mod streaming {
    pub use rig_core::streaming::*;
}

/// Portable, context-free tool contracts (used by every Rig runtime).
///
/// `Tool` is an alias for [`crate::tool::PortableTool`]: classic
/// `impl Tool for X` sites keep compiling once their `call` signature drops
/// the removed `ToolContext` parameter. Runtime-defined tools are
/// [`crate::tool::PortableDynamicTool`] records — close over your state in
/// the callback instead of threading a context. The full portable surface
/// also lives under [`crate::tool::portable`], and the same exports are at
/// [`crate::agent::tool`] for code that prefers the explicit runtime path.
/// MCP tools live in [`crate::tool::mcp`] (the `rig-mcp` crate).
pub mod tool {
    // Canonical execution values — portable, always available.
    pub use rig_core::tool::{
        IntoToolOutput, ToolErrorKind, ToolExecutionError, ToolOutput, ToolResult,
        serialize_to_tool_output,
    };
    // Runtime-independent portable contracts — explicit, always available.
    /// The classic name for the one tool-authoring trait. See the module docs
    /// for the (single) signature change relative to the removed contextual
    /// trait.
    pub use rig_core::tool::PortableTool as Tool;
    pub use rig_core::tool::{
        PortableDynamicTool, PortableTool, PortableToolEmbedding, portable_tool_definition,
    };
    // Built-in portable tools (e.g. `ThinkTool`), always available.
    pub use rig_core::tool::builtin;
    /// Session-flavoured MCP toolset: [`McpToolset`](mcp::McpToolset) pairs a
    /// [`ToolCatalog`](rig_agent::agent::prepare::ToolCatalog) with MCP-backed
    /// execution for the data-oriented runtime.
    #[cfg(all(feature = "mcp", not(target_family = "wasm")))]
    #[cfg_attr(docsrs, doc(cfg(feature = "mcp")))]
    pub use rig_mcp as mcp;
    // Runtime support the `#[derive(ToolRouter)]` expansion calls into.
    #[cfg(feature = "agent")]
    #[cfg_attr(docsrs, doc(cfg(feature = "agent")))]
    pub use rig_agent::tool::router_support;

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

/// The `#[derive(ToolRouter)]` macro: an inherent catalog/dispatch router
/// over a struct of typed tools. Requires the classic runtime
/// (`agent` feature) at expansion time.
#[cfg(all(feature = "derive", feature = "agent"))]
#[cfg_attr(docsrs, doc(cfg(all(feature = "derive", feature = "agent"))))]
pub mod tool_router {
    pub use rig_derive::ToolRouter;
}

/// Host-owned conversation memory, plus optional history-shaping policies.
///
/// Memory is not an agent slot: the host loads history before a run and
/// appends the run's committed transcript afterwards (see the
/// `rig_agent::agent_api` module docs for the exact recipe and failure
/// semantics).
///
/// This module is always available and re-exports the concrete in-process
/// store (`InMemoryConversationMemory`) and `MemoryError` from
/// `rig_core::memory`. Enabling the `memory` feature additionally re-exports
/// the `rig-memory` companion crate's policy data — `MemoryPolicy`,
/// `TokenCounter`, `Compactor`, and the concrete `PolicyMemory` whose
/// `append` returns an `AppendOutcome { stored, demoted, compaction }` — into
/// this same module.
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
