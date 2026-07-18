//! The `rig` prelude.
//!
//! Bringing this module into scope with `use rig::prelude::*` pulls in the
//! everyday portable API surface — provider-client traits, completion models,
//! embeddings, tools, and vector-store querying. Agent-runtime conveniences
//! are exported by `rig-agent::prelude` and the root `rig::prelude`.
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

// Completion model and portable core types.
pub use crate::completion::{CompletionError, CompletionModel, Message};

// Embeddings. `Embed` is re-exported from the crate root so that, with the
// `derive` feature enabled, the `#[derive(Embed)]` macro comes along with the
// trait of the same name.
pub use crate::Embed;
pub use crate::embeddings::{EmbeddingModel, EmbeddingsBuilder};

// Tools.
pub use crate::tool::Tool;

// Vector stores.
pub use crate::vector_store::VectorStoreIndex;
pub use crate::vector_store::in_memory_store::InMemoryVectorStore;
pub use crate::vector_store::request::VectorSearchRequest;

// Common container type.
pub use crate::OneOrMany;
