#![cfg_attr(docsrs, feature(doc_cfg))]
//! Public facade for Rig.
//!
//! The `rig` crate is the user-facing entry point for Rig. It re-exports the
//! portable API of `rig_core` together with the selected runtime APIs.
//!
//! # Runtime features
//!
//! The default `agent` feature selects the classic `rig-agent` runtime and
//! keeps existing `rig::prelude::*`, `.agent()`, prompt, hook, and contextual
//! tool behavior. The opt-in `bevy` feature exposes the experimental native ECS
//! runtime under [`bevy`] without adding it to the default prelude. Disabling
//! default features selects neither runtime; core-only applications should
//! normally depend directly on `rig-core`.
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
//! contract crate, including provider abstractions, built-in core
//! providers, tools, memory traits, and vector-store traits, without the root
//! facade's companion integration feature surface.

#[cfg(feature = "audio")]
pub use rig_core::audio_generation;
#[cfg(feature = "image")]
pub use rig_core::image_generation;
#[cfg(any(test, feature = "test-utils"))]
pub use rig_core::test_utils;
#[cfg(feature = "derive")]
pub use rig_core::tool_macro;
pub use rig_core::{
    Embed, EmptyListError, OneOrMany, ProviderResponseError, http_client, id, loaders, markers,
    message, model, one_or_many, providers, rerank, schemars, telemetry, transcription,
    vector_store, wasm_compat,
};

/// Provider clients and portable model constructors.
pub mod client {
    pub use rig_core::client::*;

    #[cfg(feature = "agent")]
    pub use rig_agent::client::AgentModelExt;

    /// Facade completion-client extension combining portable model construction
    /// with classic runtime constructors.
    #[cfg(feature = "agent")]
    pub trait CompletionClient: rig_core::client::CompletionClient + Sized {
        /// Construct a portable completion model.
        fn completion_model(
            &self,
            model: impl Into<String>,
        ) -> <Self as rig_core::client::CompletionClient>::CompletionModel {
            rig_core::client::CompletionClient::completion_model(self, model)
        }

        /// Construct a classic agent builder.
        fn agent(
            &self,
            model: impl Into<String>,
        ) -> rig_agent::agent::AgentBuilder<
            <Self as rig_core::client::CompletionClient>::CompletionModel,
        > {
            rig_agent::client::AgentClientExt::agent(self, model)
        }

        /// Construct a classic structured extractor builder.
        fn extractor<T>(
            &self,
            model: impl Into<String>,
        ) -> rig_agent::extractor::ExtractorBuilder<
            <Self as rig_core::client::CompletionClient>::CompletionModel,
            T,
        >
        where
            T: rig_core::schemars::JsonSchema
                + for<'de> serde::Deserialize<'de>
                + serde::Serialize
                + rig_core::wasm_compat::WasmCompatSend
                + rig_core::wasm_compat::WasmCompatSync
                + 'static,
        {
            rig_agent::client::AgentClientExt::extractor(self, model)
        }
    }

    #[cfg(feature = "agent")]
    impl<C> CompletionClient for C where C: rig_core::client::CompletionClient {}
}

/// Portable completion contracts plus classic prompting conveniences when enabled.
pub mod completion {
    pub use rig_core::completion::*;

    #[cfg(feature = "agent")]
    pub use rig_agent::completion::{
        Chat, Prompt, PromptError, StructuredOutputError, TypedPrompt,
    };
}

/// Portable embedding contracts plus classic runtime embedding tools when enabled.
pub mod embeddings {
    pub use rig_core::embeddings::*;

    #[cfg(feature = "agent")]
    pub use rig_agent::embeddings::{ToolSchema, tool};
}

/// Portable stream values plus classic high-level streaming traits when enabled.
pub mod streaming {
    pub use rig_core::streaming::*;

    #[cfg(feature = "agent")]
    pub use rig_agent::streaming::{StreamingChat, StreamingPrompt};
}

/// Portable tool contracts plus classic contextual dispatch when enabled.
pub mod tool {
    pub use rig_core::tool::*;

    #[cfg(feature = "agent")]
    pub use rig_agent::tool::server::{ToolServer, ToolServerHandle};
    #[cfg(feature = "agent")]
    pub use rig_agent::tool::{
        ContextualTool, MissingToolContext, ToolContext, ToolEmbedding, ToolSet,
    };

    /// Classic tool-server APIs.
    #[cfg(feature = "agent")]
    pub mod server {
        pub use rig_agent::tool::server::*;
    }

    /// Classic Model Context Protocol integration.
    #[cfg(feature = "rmcp")]
    pub mod rmcp {
        pub use rig_agent::tool::rmcp::*;
    }
}

/// The classic Rig agent runtime.
#[cfg(feature = "agent")]
#[cfg_attr(docsrs, doc(cfg(feature = "agent")))]
pub mod agent {
    pub use rig_agent::agent::*;
    pub use rig_agent::client::{AgentClientExt, AgentModelExt};
}

#[cfg(feature = "agent")]
pub use rig_agent::{ExtractionResponse, extractor, integrations};

/// The experimental ECS-native Rig runtime.
#[cfg(feature = "bevy")]
#[cfg_attr(docsrs, doc(cfg(feature = "bevy")))]
pub mod bevy {
    pub use rig_bevy::*;
}

/// Common imports for the portable core and the default classic runtime.
pub mod prelude {
    pub use rig_core::prelude::*;

    #[cfg(feature = "agent")]
    pub use crate::{
        agent::{Agent, AgentModelExt, MultiTurnStreamItem, StreamingResult},
        client::CompletionClient,
        completion::{Chat, Prompt, PromptError, StructuredOutputError, TypedPrompt},
        streaming::{StreamingChat, StreamingPrompt},
        tool::{ContextualTool, ToolSet},
    };
}

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
