//! Common imports for Rig's classic runtime.

pub use rig_core::client::verify::VerifyError;

pub use crate::agent::{
    Agent, AgentHook, HookContext, ModelHandle, ModelRef, ModelSelection, ModelSelectionAction,
    MultiTurnStreamItem, RunEvents, StreamingResult,
};
pub use crate::client::{AgentModelExt, AgentProviderExt};
pub use crate::completion::{
    CompletionError, CompletionModel, Message, PromptError, StructuredOutputError,
};
pub use crate::tool::{Tool, ToolSet};
pub use rig_core::driver::{Bind, Bound, CompletionProvider};

pub use rig_core::Embed;
pub use rig_core::embeddings::{EmbeddingModel, EmbeddingsBuilder};
pub use rig_core::vector_store::VectorStoreIndex;
pub use rig_core::vector_store::in_memory_store::InMemoryVectorStore;
pub use rig_core::vector_store::request::VectorSearchRequest;
