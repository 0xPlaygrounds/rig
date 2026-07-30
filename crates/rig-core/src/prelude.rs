//! The `rig` prelude.
//!
//! Bringing this module into scope with `use rig::prelude::*` pulls in the
//! data types that the provider `functions` modules speak: completion requests
//! and responses, embeddings, tool and vector-store contracts, plus the
//! provider-configuration vocabulary ([`ProviderDescriptor`], [`ApiKeyLocation`],
//! [`ConfigError`]) and the shared [`HttpRuntime`] transport.
//!
//! There are no client or model *traits* to import: each provider exposes a
//! plain `functions::Config` and free functions taking
//! `(&Config, &HttpRuntime, request)`, so calling a provider is a matter of
//! naming its module rather than bringing a trait into scope.
//!
//! This is deliberately the *common* path, not the whole crate. Advanced
//! surfaces — the hook system, the run-loop stepping types, message content
//! blocks, tool authoring internals, extraction/loaders/memory, etc. — are
//! imported explicitly from their modules so those imports document intent.

// Provider configuration and transport.
pub use crate::http_runtime::HttpRuntime;
pub use crate::providers::{ApiKeyLocation, ConfigError, ProviderDescriptor};

// Completion.
pub use crate::completion::{CompletionError, CompletionRequest, CompletionResponse, Message};

// Embeddings. `Embed` is re-exported from the crate root so that, with the
// `derive` feature enabled, the `#[derive(Embed)]` macro comes along with the
// trait of the same name.
pub use crate::Embed;
pub use crate::embeddings::{Embedding, EmbeddingError, embed_documents};

// Tools.
pub use crate::tool::PortableTool;

// Vector stores.
pub use crate::vector_store::in_memory_store::InMemoryVectorStore;
pub use crate::vector_store::request::VectorSearchRequest;
pub use crate::vector_store::{SearchHit, StoreRecord};

// Common container type.
pub use crate::OneOrMany;
