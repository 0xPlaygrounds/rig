//! The `rig` prelude.
//!
//! Bringing this module into scope with `use rig::prelude::*` pulls in the
//! everyday API surface — the provider-client traits, agents, prompting and
//! chatting (blocking and streaming), embeddings, tools, and vector-store
//! querying — so a basic agent or RAG program only needs to additionally import
//! its provider module (`rig::providers::…`).
//!
//! This is deliberately the *common* path, not the whole crate. Advanced
//! surfaces — the hook system, the run-loop stepping types, message content
//! blocks, tool authoring internals, extraction/loaders/memory, etc. — are
//! imported explicitly from their modules so those imports document intent.

// Provider-client traits.
pub use crate::client::ProviderClient;
pub use crate::client::completion::CompletionClient;
pub use crate::client::embeddings::EmbeddingsClient;
pub use crate::client::model_listing::ModelListingClient;
pub use crate::client::transcription::TranscriptionClient;
pub use crate::client::verify::{VerifyClient, VerifyError};

#[cfg(feature = "image")]
pub use crate::client::image_generation::ImageGenerationClient;

#[cfg(feature = "audio")]
pub use crate::client::audio_generation::AudioGenerationClient;

// Agents.
pub use crate::agent::Agent;

// Completion: prompting, chatting, the model trait, and the core types.
pub use crate::completion::{
    Chat, CompletionError, CompletionModel, Message, Prompt, PromptError, StructuredOutputError,
    TypedPrompt,
};

// Streaming counterparts of the blocking `Prompt`/`Chat` traits, plus the items
// yielded when consuming an agent stream.
pub use crate::agent::{MultiTurnStreamItem, StreamingResult};
pub use crate::streaming::{StreamingChat, StreamingPrompt};

// Embeddings. `Embed` is re-exported from the crate root so that, with the
// `derive` feature enabled, the `#[derive(Embed)]` macro comes along with the
// trait of the same name.
pub use crate::Embed;
pub use crate::embeddings::{EmbeddingModel, EmbeddingsBuilder};

// Tools.
pub use crate::tool::{Tool, ToolSet};

// Vector stores.
pub use crate::vector_store::VectorStoreIndex;
pub use crate::vector_store::in_memory_store::InMemoryVectorStore;
pub use crate::vector_store::request::VectorSearchRequest;

// Common container type.
pub use crate::OneOrMany;
