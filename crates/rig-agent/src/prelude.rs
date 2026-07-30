//! Common imports for Rig's classic runtime.

pub use rig_core::client::ProviderClient;
pub use rig_core::client::embeddings::EmbeddingsClient;
pub use rig_core::client::model_listing::ModelListingClient;
pub use rig_core::client::transcription::TranscriptionClient;
pub use rig_core::client::verify::{VerifyClient, VerifyError};

#[cfg(feature = "audio")]
pub use rig_core::client::audio_generation::AudioGenerationClient;
#[cfg(feature = "image")]
pub use rig_core::client::image_generation::ImageGenerationClient;

pub use crate::agent::{Agent, PromptResponse, SessionRunner};
pub use crate::client::{AgentClientExt, ToProviderConfig};
pub use crate::completion::{
    CompletionError, CompletionModel, Message, PromptError, StructuredOutputError,
};
pub use crate::stream::{AgentStream, AgentStreamItem};
pub use crate::tool::{PortableDynamicTool, PortableTool};
pub use rig_core::client::completion::CompletionClient;

pub use rig_core::Embed;
pub use rig_core::OneOrMany;
pub use rig_core::embeddings::{EmbeddingModel, EmbeddingsBuilder};
pub use rig_core::vector_store::in_memory_store::InMemoryVectorStore;
pub use rig_core::vector_store::request::VectorSearchRequest;
