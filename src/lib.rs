#![cfg_attr(docsrs, feature(doc_cfg))]
//! Public facade for Rig.
//!
//! Re-exports `rig_core` at `rig::...` paths and, under the default `agent`
//! feature, the runtime from `rig_agent` at `rig::agent`. `rig::tool` then
//! carries the contextual tool API alongside the portable contracts, which are
//! always available. `use rig::prelude::*;` brings in [`Model`], its
//! [`Transport`](rig_core::driver::Transport), [`model`] and the common
//! agent types.
//!
//! Companion provider and vector-store crates are feature-gated modules, named
//! after their features wherever module naming allows:
//!
//! ```toml
//! [dependencies]
//! rig = { version = "*", features = ["lancedb", "fastembed"] }
//! ```
//!
//! Depend on `rig-core` directly to skip this facade's companion integrations.

pub use rig_core::*;

/// The bundled `reqwest` transport (`rig-reqwest`). A provider's wire and a
/// transport make a [`Model`]; [`model`] pairs a wire with the process-wide
/// default transport ([`rig_reqwest::shared`]). Without the feature, pair a
/// wire with any `HttpClientExt` implementation through `Model::new`.
#[cfg(feature = "reqwest")]
#[cfg_attr(docsrs, doc(cfg(feature = "reqwest")))]
pub use rig_reqwest;

/// `wire` on the process-wide default transport: the bundled reqwest client,
/// built once on first use and shared by every model made here.
///
/// Construction never fails. When the reqwest client cannot be built, every
/// call on the model reports the build failure as
/// [`ProviderError::Http`](rig_core::error::ProviderError::Http).
///
/// ```no_run
/// use rig::providers::openai::{self, OpenAI};
///
/// # fn main() -> Result<(), rig::client::EnvError> {
/// let model = rig::model(OpenAI::from_env()?.completion(openai::GPT_5_2));
/// # let _ = model;
/// # Ok(())
/// # }
/// ```
#[cfg(feature = "reqwest")]
#[cfg_attr(docsrs, doc(cfg(feature = "reqwest")))]
pub fn model<W: rig_core::wire::Wire>(wire: W) -> Model<W> {
    Model::new(wire, rig_reqwest::shared())
}

/// The bundled `tokio-tungstenite` websocket backend and its default-backend
/// conveniences (`rig-tungstenite`), on native targets. With the `websocket`
/// feature, `model.responses_websocket()` opens a session over it with no
/// backend named; without it, rig has no websocket backend and a session is
/// opened with `responses_websocket_with(..)` and any
/// [`rig_core::ws_client::WebSocketClientExt`] implementation.
#[cfg(all(feature = "websocket", not(target_family = "wasm")))]
#[cfg_attr(docsrs, doc(cfg(feature = "websocket")))]
pub use rig_tungstenite;

/// Provider configurations and their wires. A wire says what to send and how
/// to read the reply; pair it with a transport in a [`Model`] to call it.
pub mod providers {
    pub use rig_core::providers::*;
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
pub use rig_agent::{Agent, AgentBuilder, AgentRun, AgentRunner, TypedPromptResponse};

/// Direct access to the portable provider and data contracts.
pub mod core {
    pub use rig_core::*;
}

/// The classic runtime's effect bus (`rig_agent::bus`): the dispatcher,
/// the registrar, the driver, the typed views. What a handler implements is
/// `rig::core::serve`; the vocabulary is `rig::core::effect`. Effects
/// without the classic agent are `rig-ecs`'s.
#[cfg(feature = "agent")]
#[cfg_attr(docsrs, doc(cfg(feature = "agent")))]
pub mod bus {
    pub use rig_agent::bus::*;
}

/// Effect-log recording and replay; optional runtime and native HTTP integrations.
pub use rig_cassette as cassette;

/// The sans-IO run layer of rig-agent (`rig_agent::run`): `AgentRun` and its
/// step/turn types, the run's spec, request preparation, output policy and
/// per-turn patch, its response and error types, the invalid-call decision
/// data, the streamed-turn assembler and the loop-side transcript helpers.
/// rig-core keeps only the message-model invariants (`rig::core::transcript`).
#[cfg(feature = "agent")]
#[cfg_attr(docsrs, doc(cfg(feature = "agent")))]
pub mod run {
    pub use rig_agent::run::*;
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

/// Environment handling for provider configurations.
pub mod client {
    pub use rig_core::client::*;
}

/// Low-level completion contracts plus the classic runtime's errors.
pub mod completion {
    #[cfg(feature = "agent")]
    pub use rig_agent::completion::{PromptError, StructuredOutputError};
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
    // The contextual `Tool` and its mutable `ToolContext`.
    #[cfg(feature = "reqwest")]
    pub use crate::model;
    #[cfg(feature = "agent")]
    pub use crate::tool::{Tool, ToolContext};
    #[cfg(feature = "agent")]
    pub use rig_agent::prelude::{
        Agent, AgentBuilder, MultiTurnStreamItem, PromptError, RunEvents, StreamingResult,
        StructuredOutputError, ToolSet,
    };
    pub use rig_core::prelude::*;
    // Default-backend websocket traits: `client.responses_websocket(..)` and
    // `builder().connect()` over the bundled tungstenite backend, plus the
    // provider's own session extension trait.
    #[cfg(all(feature = "websocket", not(target_family = "wasm")))]
    pub use rig_tungstenite::prelude::*;
}

/// Low-level streaming values.
pub mod streaming {
    pub use rig_core::streaming::*;
}

/// Tools: contextual authoring, the erased tool set, and the live registry.
///
/// The contextual and portable contracts come from `rig-core` and need no
/// feature. The registry, tool set, and catalog belong to the agent runtime and
/// need the `agent` feature; they also live at [`crate::agent::tool`].
pub mod tool {
    /// Derive a stable serialized key for a tool-context value.
    #[cfg(feature = "derive")]
    #[cfg_attr(docsrs, doc(cfg(feature = "derive")))]
    pub use rig_derive::ContextValue;

    #[cfg(feature = "agent")]
    #[cfg_attr(docsrs, doc(cfg(feature = "agent")))]
    pub use rig_agent::tool::{
        RegisteredTool, ToolCatalog, ToolDispatch, ToolLease, ToolSet, execute_tool,
    };
    pub use rig_core::tool::builtin;
    pub use rig_core::tool::{
        ContextValue, DynamicTool, ErasedTool, Tool, ToolContext, ToolContextError, ToolEmbedding,
        tool_definition,
    };
    pub use rig_core::tool::{
        IntoToolOutput, ToolErrorKind, ToolExecutionError, ToolOutput, ToolResult,
    };
    pub use rig_core::tool::{PortableTool, PortableToolEmbedding};

    /// MCP tool support from `rig-rmcp`, which supports native targets only.
    #[cfg(all(feature = "rmcp", not(target_family = "wasm")))]
    #[cfg_attr(docsrs, doc(cfg(feature = "rmcp")))]
    pub mod rmcp {
        pub use rig_rmcp::*;
    }
    /// The live registry, layered over the contracts above.
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

/// Conversation memory traits and the in-process backend, plus the `rig-memory`
/// policy types when the `memory` feature is enabled.
pub mod memory {
    pub use rig_core::memory::*;

    #[cfg(feature = "memory")]
    #[cfg_attr(docsrs, doc(cfg(feature = "memory")))]
    pub use rig_memory::*;
}

/// Declares one feature-gated facade module per companion crate, each row giving
/// the module, its crate, and the features enabling it.
///
/// The single-feature arm is separate so rustdoc renders "Available on crate
/// feature `x` only" rather than a one-element `any`.
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
    /// Typed TypeSafe Jev judgments for routing and rubric evaluation.
    typesafeai = rig_typesafeai ["typesafeai"];
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
