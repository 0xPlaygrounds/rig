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
//! providers, tools, memory traits, and vector-store traits, without the root
//! facade's companion integration feature surface.

pub use rig_core::*;

/// The bundled `reqwest` transport and its default-transport conveniences
/// (`rig-reqwest`). With the default `reqwest` feature, [`providers`] is the
/// aliased tree whose types default to [`rig_reqwest::ReqwestClient`], and
/// [`prelude`] carries [`rig_reqwest::client::DefaultTransportClient`] /
/// [`rig_reqwest::client::DefaultTransportBuilder`]. Without it, rig has no
/// default transport: construct clients with `new_with(..)` / `.http_client(..)`
/// and any `HttpClientExt` implementation.
#[cfg(feature = "reqwest")]
#[cfg_attr(docsrs, doc(cfg(feature = "reqwest")))]
pub use rig_reqwest;

/// The bundled `tokio-tungstenite` websocket backend and its default-backend
/// conveniences (`rig-tungstenite`), on native targets. With the `websocket`
/// feature,
/// `client.responses_websocket("gpt-5.4")` opens a session over it with no
/// backend named; without it, rig has no websocket backend and a session is
/// opened with `connect_with(..)` and any
/// [`rig_core::ws_client::WebSocketClientExt`] implementation.
#[cfg(all(feature = "websocket", not(target_family = "wasm")))]
#[cfg_attr(docsrs, doc(cfg(feature = "websocket")))]
pub use rig_tungstenite;

/// Provider clients and models, with the transport defaulted to the bundled
/// `reqwest` one.
#[cfg(feature = "reqwest")]
#[cfg_attr(docsrs, doc(cfg(feature = "reqwest")))]
pub mod providers {
    pub use rig_reqwest::providers::*;
}

/// Transport-agnostic HTTP contracts, plus the bundled reqwest transport type.
pub mod http_client {
    pub use rig_core::http_client::*;
    #[cfg(feature = "reqwest")]
    #[cfg_attr(docsrs, doc(cfg(feature = "reqwest")))]
    pub use rig_reqwest::{ReqwestClient, from_reqwest};
}

#[cfg(feature = "agent")]
#[cfg_attr(docsrs, doc(cfg(feature = "agent")))]
pub use rig_agent::{Agent, AgentBuilder, AgentRun, AgentRunner, ExtractionResponse};

/// Direct access to the portable provider and data contracts.
pub mod core {
    pub use rig_core::*;
}

/// The sans-IO agent-run protocol (`rig-run`): `AgentRun` and the data a
/// driver needs to step it. Available without the classic runtime, so a host
/// that drives runs itself (an ECS plugin, a job system) does not need `agent`.
pub mod run {
    pub use rig_run::*;
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

/// Provider clients plus classic agent/extractor constructors.
pub mod client {
    // Classic-runtime construction extensions: `agent()` / `extractor()` on any
    // completion client (`AgentClientExt`) and `into_agent_builder()` on any
    // completion model (`AgentModelExt`).
    #[cfg(feature = "agent")]
    pub use rig_agent::client::{AgentClientExt, AgentModelExt};

    // The full portable provider-client surface, including the canonical
    // `CompletionClient`. `AgentClientExt` is a distinct name, so there is no
    // shadow — just one canonical completion-client trait plus the classic
    // construction extension.
    pub use rig_core::client::*;

    // Default-transport construction (`Client::new(key)`, `from_env()`,
    // `builder().…build()`) over the bundled reqwest transport.
    #[cfg(feature = "reqwest")]
    #[cfg_attr(docsrs, doc(cfg(feature = "reqwest")))]
    pub use rig_reqwest::client::{DefaultTransportBuilder, DefaultTransportClient};
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
    // The classic construction extension `AgentClientExt` (adding `agent()` /
    // `extractor()`) sits alongside the canonical `CompletionClient` brought in
    // by the `rig_core::prelude::*` glob below. The two traits share no method
    // names, so both resolve without ambiguity and together restore the
    // pre-split `client.completion_model(m)` / `client.agent(m)` surface.
    #[cfg(feature = "agent")]
    pub use rig_agent::prelude::{
        Agent, AgentClientExt, AgentModelExt, Chat, MultiTurnStreamItem, Prompt, PromptError,
        RunEvents, StreamingChat, StreamingPrompt, StreamingResult, StructuredOutputError, ToolSet,
        TypedPrompt,
    };
    pub use rig_core::prelude::*;
    // Default-transport construction traits: `Client::new(..)` / `from_env()` /
    // `builder().build()` over the bundled reqwest transport.
    #[cfg(feature = "reqwest")]
    pub use rig_reqwest::prelude::*;
    // Default-backend websocket traits: `client.responses_websocket(..)` and
    // `builder().connect()` over the bundled tungstenite backend, plus the
    // provider's own session extension trait.
    #[cfg(all(feature = "websocket", not(target_family = "wasm")))]
    pub use rig_tungstenite::prelude::*;
}

/// Low-level streaming values plus classic streaming traits.
pub mod streaming {
    #[cfg(feature = "agent")]
    pub use rig_agent::streaming::{StreamingChat, StreamingPrompt};
    pub use rig_core::streaming::*;
}

/// Tools: contextual authoring, the erased tool set, and the live registry.
///
/// `Tool`, `ToolContext`, `DynamicTool`, `ToolSet`, and `ToolCatalog` are
/// rig-core types, available with or without the `agent` feature, so
/// `use rig::tool::{Tool, ToolContext};` keeps working everywhere. The
/// runtime-independent portable contracts are exposed explicitly as
/// [`crate::tool::PortableTool`], [`crate::tool::PortableToolEmbedding`], and
/// [`crate::tool::PortableDynamicTool`] (and in full under
/// [`crate::tool::portable`]). The live registry (`server`) is the agent
/// runtime's and needs the `agent` feature; the same surface also lives at
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
    // Contextual authoring and the erased tool set — rig-core, always available.
    pub use rig_core::tool::{
        DynamicTool, ErasedTool, MissingToolContext, RegisteredTool, Tool, ToolCatalog,
        ToolContext, ToolDispatch, ToolEmbedding, ToolSet, dispatch_tool, tool_definition,
    };
    // Built-in portable tools (e.g. `ThinkTool`), always available.
    pub use rig_core::tool::builtin;

    // MCP tool support from the companion `rig-rmcp` crate (rig-core only;
    // native-only: the crate root raises a `compile_error!` on wasm, which CI
    // asserts is the only error). Kept at `rig::tool::rmcp` so existing paths
    // resolve. rig-agent's `ToolServerHandle` implements the
    // `ManagedToolSink` its `McpClientHandler` registers into.
    #[cfg(all(feature = "rmcp", not(target_family = "wasm")))]
    #[cfg_attr(docsrs, doc(cfg(feature = "rmcp")))]
    pub mod rmcp {
        pub use rig_rmcp::*;
    }
    // The live registry (`ToolServer`/`ToolServerHandle`): retrieval indexes,
    // managed remote tool sources, and the per-turn snapshot — the agent
    // runtime's, layered over the rig-core types above.
    #[cfg(feature = "agent")]
    #[cfg_attr(docsrs, doc(cfg(feature = "agent")))]
    pub use rig_agent::tool::server;

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

/// Declare one feature-gated facade module per companion crate.
///
/// Each row expands to the same four items — the `cfg` gate, the matching
/// docs.rs `doc(cfg)` note, the module, and its glob re-export. A module
/// declaration cannot come from a function or a trait, so a macro is the only
/// way to state that shape once instead of per crate; the rows keep the
/// module → crate → feature mapping readable as a table.
///
/// The single- and multi-feature arms are separate on purpose: a one-feature
/// module must render rustdoc's "Available on crate feature `x` only", which
/// `doc(cfg(any(feature = "x")))` would spell as a one-element `any`.
macro_rules! companion_modules {
    () => {};
    (
        $(#[doc = $doc:literal])*
        $module:ident = $krate:ident [$feature:literal];
        $($rest:tt)*
    ) => {
        $(#[doc = $doc])*
        #[cfg(feature = $feature)]
        #[cfg_attr(docsrs, doc(cfg(feature = $feature)))]
        pub mod $module {
            pub use $krate::*;
        }
        companion_modules! { $($rest)* }
    };
    (
        $(#[doc = $doc:literal])*
        $module:ident = $krate:ident [$($feature:literal),+ $(,)?];
        $($rest:tt)*
    ) => {
        $(#[doc = $doc])*
        #[cfg(any($(feature = $feature),+))]
        #[cfg_attr(docsrs, doc(cfg(any($(feature = $feature),+))))]
        pub mod $module {
            pub use $krate::*;
        }
        companion_modules! { $($rest)* }
    };
}

companion_modules! {
    bedrock = rig_bedrock ["bedrock"];
    /// Local CPU inference with validated Llama/SmolLM2 and native tool-capable Qwen3 models.
    candle = rig_candle ["candle"];
    fastembed = rig_fastembed [
        "fastembed",
        "fastembed-hf-hub",
        "fastembed-ort-download-binaries",
    ];
    gemini_grpc = rig_gemini_grpc ["gemini-grpc"];
    helixdb = rig_helixdb ["helixdb"];
    lancedb = rig_lancedb ["lancedb"];
    milvus = rig_milvus ["milvus"];
    mongodb = rig_mongodb ["mongodb"];
    neo4j = rig_neo4j ["neo4j"];
    postgres = rig_postgres ["postgres"];
    qdrant = rig_qdrant ["qdrant"];
    s3vectors = rig_s3vectors ["s3vectors"];
    scylladb = rig_scylladb ["scylladb"];
    sqlite = rig_sqlite ["sqlite"];
    surrealdb = rig_surrealdb ["surrealdb"];
    vectorize = rig_vectorize ["vectorize"];
    vertexai = rig_vertexai ["vertexai"];
}
