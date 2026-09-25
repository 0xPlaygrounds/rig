//! Common imports for Rig's classic runtime.
//!
//! ```
//! use rig_agent::prelude::*;
//! use rig_core::operation::Completion;
//!
//! fn assistant(model: impl Into<BoxedModel<Completion>>) -> Agent {
//!     AgentBuilder::new(model).build()
//! }
//! ```

pub use crate::agent::{
    Agent, AgentBuilder, AgentHook, HookContext, ModelHandle, ModelRef, ModelSelection,
    ModelSelectionAction, MultiTurnStreamItem, RunEvents, StreamingResult,
};
pub use crate::completion::{Message, PromptError, StructuredOutputError};
pub use crate::tool::{Tool, ToolSet};
pub use rig_core::driver::{BoxedModel, Model, Transport};
pub use rig_core::error::ProviderError;
pub use rig_core::wire::Wire;

pub use rig_core::Embed;
pub use rig_core::embeddings::EmbeddingsBuilder;
pub use rig_core::vector_store::VectorStoreIndex;
pub use rig_core::vector_store::in_memory_store::InMemoryVectorStore;
pub use rig_core::vector_store::request::VectorSearchRequest;
