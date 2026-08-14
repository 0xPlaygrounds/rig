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
    // Native-only, matching every other `rmcp` gate: the module does not exist
    // on wasm. Reaching this re-export there needs `rig-agent` to have compiled
    // first, which its own `compile_error!` prevents, so the predicate is
    // belt-and-braces — but the CI error-count assertion only builds
    // `-p rig-agent`, so nothing else would catch this one drifting.
    #[cfg(all(feature = "agent", feature = "rmcp", not(target_family = "wasm")))]
    #[cfg_attr(docsrs, doc(cfg(feature = "rmcp")))]
    pub use rig_agent::tool::rmcp;
    #[cfg(feature = "agent")]
    #[cfg_attr(docsrs, doc(cfg(feature = "agent")))]
    pub use rig_agent::tool::{
        DynamicTool, MissingToolContext, Tool, ToolContext, ToolEmbedding, ToolSet, server,
        tool_definition,
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
