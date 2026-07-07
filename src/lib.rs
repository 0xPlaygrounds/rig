#![cfg_attr(docsrs, feature(doc_cfg))]
//! Public facade for Rig.
//!
//! The `rig` crate is the user-facing entry point for Rig. It re-exports the
//! full public API of `rig_core`, so core traits, builders, providers, tools,
//! and request/response types are available through `rig::...` paths.
//!
//! # Companion integrations
//!
//! Companion provider crates are exposed as feature-gated modules on this
//! facade. Enable only the integrations your application uses:
//!
//! ```toml
//! [dependencies]
//! rig = { version = "0.36.0", features = ["bedrock", "fastembed"] }
//! ```
//!
//! This enables modules such as `rig::bedrock` and `rig::fastembed`. Other
//! companion integrations follow the same pattern, with feature names aligned to
//! their facade module paths wherever Rust module naming allows it.
//!
//! # Retrieval (RAG)
//!
//! Rig does not ship a built-in vector-store abstraction. Retrieval is a
//! user-land pattern: expose it as an ordinary tool the model calls, or inject
//! retrieved context before each model call from an
//! [`AgentHook`](rig_core::agent::AgentHook) via
//! [`RequestPatch::extra_context`](rig_core::agent::RequestPatch). Embedding
//! models/builders remain in [`rig_core::embeddings`]. See the `tool_active_rag`
//! and `hook_passive_rag` examples.
//!
//! # When to use `rig-core` directly
//!
//! Depend on the `rig-core` package directly when you only need the core Rig
//! implementation crate, including provider abstractions, built-in core
//! providers, tools, and memory traits, without the root facade's companion
//! integration feature surface.

pub use rig_core::*;

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

#[cfg(feature = "vertexai")]
#[cfg_attr(docsrs, doc(cfg(feature = "vertexai")))]
pub mod vertexai {
    pub use rig_vertexai::*;
}
