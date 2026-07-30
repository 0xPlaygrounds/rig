//! Common imports for building and driving agents.
//!
//! Providers are configured as plain data — `<provider>::functions::Config` —
//! and selected through [`ProviderConfig`](crate::provider::ProviderConfig);
//! there is no client type and no capability trait to import.

pub use crate::agent::{Agent, AgentBuilder, PromptResponse, SessionRunner};
pub use crate::completion::{CompletionError, Message, PromptError, StructuredOutputError};
pub use crate::provider::{EmbedderConfig, ProviderConfig, Runtime};
pub use crate::stream::{AgentStream, AgentStreamItem};
pub use crate::tool::{PortableDynamicTool, PortableTool};

pub use rig_core::Embed;
pub use rig_core::OneOrMany;
pub use rig_core::providers::ConfigError;
pub use rig_core::vector_store::in_memory_store::InMemoryVectorStore;
pub use rig_core::vector_store::request::VectorSearchRequest;
