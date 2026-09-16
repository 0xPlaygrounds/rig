//! The `rig` prelude.
//!
//! Bringing this module into scope with `use rig::prelude::*` pulls in the
//! portable model-construction, completion, embedding, tool, and vector-store
//! contracts.
//!
//! This is deliberately the *common* path, not the whole crate. Advanced
//! surfaces — the hook system, the run-loop stepping types, message content
//! blocks, tool authoring internals, extraction/loaders/memory, etc. — are
//! imported explicitly from their modules so those imports document intent.

// The `Verify` operation's error, returned by `Bound::verify`.
pub use crate::client::verify::VerifyError;

pub use crate::completion::{CompletionError, CompletionModel, Message};
// Construction: a wire bound to a socket is the model, and anything that
// builds a completion model answers `completion(model)` — including the
// typed-transport providers, which are not wires.
pub use crate::driver::{Bind, Bound, CompletionProvider};

// Embeddings. `Embed` is re-exported from the crate root so that, with the
// `derive` feature enabled, the `#[derive(Embed)]` macro comes along with the
// trait of the same name.
pub use crate::Embed;
pub use crate::embeddings::{EmbeddingModel, EmbeddingsBuilder};

// Tools.
pub use crate::tool::PortableTool;

// Vector stores.
pub use crate::vector_store::VectorStoreIndex;
pub use crate::vector_store::in_memory_store::InMemoryVectorStore;
pub use crate::vector_store::request::VectorSearchRequest;
