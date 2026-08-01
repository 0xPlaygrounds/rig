//! Common imports for building and driving agents.
//!
//! Provider operation data remains available as
//! `<provider>::functions::Config` and [`ProviderConfig`]. Concrete,
//! monomorphic provider clients add fluent agent and bound-completion methods
//! through the extension traits exported here.

pub use crate::agent::{Agent, AgentBuilder, PromptResponse, SessionRunner};
pub use crate::client::{
    AgentClientExt, BindCompletionExt, BoundCompletionRequest, CompletionClientExt,
    CompletionHandle, ToProviderConfig,
};
pub use crate::completion::{CompletionError, Message, PromptError, StructuredOutputError};
pub use crate::provider::{EmbedderConfig, ProviderConfig, Runtime};
pub use crate::stream::{AgentStream, AgentStreamItem};
pub use crate::tool::{PortableDynamicTool, PortableTool};

pub use rig_core::Embed;
pub use rig_core::OneOrMany;
pub use rig_core::providers::ConfigError;
pub use rig_core::vector_store::in_memory_store::InMemoryVectorStore;
pub use rig_core::vector_store::request::VectorSearchRequest;
