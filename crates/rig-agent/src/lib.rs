//! The classic Rig agent runtime.
//!
//! This crate owns agent orchestration, hooks, prompting, tool dispatch, and
//! runtime lifecycle behavior. Provider contracts and canonical values remain
//! in `rig-core`.

#![cfg_attr(
    test,
    allow(
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::unwrap_used,
        clippy::unreachable
    )
)]

pub use rig_core::{
    OneOrMany, client, embeddings, id, markers, memory, message, model, one_or_many, providers,
    telemetry, vector_store, wasm_compat,
};

pub mod json_utils {
    pub use rig_core::json_utils::*;
}

pub mod agent;
pub mod client_ext;
pub mod completion;
pub mod extractor;
pub mod integrations;
pub mod streaming;
pub mod tool;

#[cfg(any(test, feature = "test-utils"))]
pub mod test_utils;

/// Common imports for the classic runtime.
pub mod prelude {
    pub use crate::agent::{Agent, MultiTurnStreamItem, StreamingResult};
    pub use crate::client_ext::{AgentClientExt, AgentModelExt};
    pub use crate::completion::{Chat, Prompt, PromptError, StructuredOutputError, TypedPrompt};
    pub use crate::streaming::{StreamingChat, StreamingPrompt};
    pub use crate::tool::{ToolContext, ToolSet};
    pub use rig_core::prelude::*;
}
