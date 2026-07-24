#![cfg_attr(docsrs, feature(doc_cfg))]
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
//! Rig is a Rust library for building LLM-powered applications that focuses on ergonomics and modularity.
//!
//! # Table of contents
//! - [High-level features](#high-level-features)
//! - [Simple Example](#simple-example)
//! - [Core Concepts](#core-concepts)
//! - [Integrations](#integrations)
//!
//! # High-level features
//! - Full support for LLM completion and embedding workflows
//! - Simple but powerful common abstractions over LLM providers (e.g. OpenAI, Cohere) and vector stores (e.g. MongoDB, in-memory)
//! - Integrate LLMs in your app with minimal boilerplate
//!
//! # Simple example
//! ```no_run
//! use rig_core::{
//!     client::{CompletionClient, ProviderClient},
//!     completion::{AssistantContent, CompletionModel},
//!     providers::openai,
//! };
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create an OpenAI client and completion model.
//!     // This requires the `OPENAI_API_KEY` environment variable to be set.
//!     let openai_client = openai::Client::from_env()?;
//!     let model = openai_client.completion_model(openai::GPT_5_2);
//!
//!     let request = model.completion_request("Who are you?").build();
//!     let response = model.completion(request).await?;
//!     for item in response.choice {
//!         if let AssistantContent::Text(text) = item {
//!             println!("{}", text.text);
//!         }
//!     }
//!
//!     Ok(())
//! }
//! ```
//! Note: using `#[tokio::main]` requires you enable tokio's `macros` and `rt-multi-thread` features
//! or just `full` to enable all features (`cargo add tokio --features macros,rt-multi-thread`).
//!
//! # Core concepts
//! ## Completion and embedding models
//! Rig provides a consistent API for working with LLMs and embeddings. Specifically,
//! each provider (e.g. OpenAI, Cohere) has a `Client` struct that can be used to initialize completion
//! and embedding models. These models implement the [CompletionModel](crate::completion::CompletionModel)
//! and [EmbeddingModel](crate::embeddings::EmbeddingModel) traits respectively, which provide a common,
//! low-level interface for creating completion and embedding requests and executing them.
//!
//! ## Agent runtimes
//! This crate owns the provider-agnostic model, message, tool, and storage
//! contracts. The sibling `rig-agent` crate provides the classic builder and
//! run-loop API.
//!
//! ## Vector stores and indexes
//! Rig provides a common interface for working with vector stores and indexes. Specifically, the library
//! provides the [VectorStoreIndex](crate::vector_store::VectorStoreIndex)
//! trait, which can be implemented to define vector stores and indices respectively.
//! Indexes can be queried directly by applications or runtimes. For active RAG,
//! expose the index through its blanket [`PortableTool`](crate::tool::PortableTool)
//! implementation, or through a custom tool, so the model decides when and how
//! to retrieve. The classic `rig-agent` runtime can also query indexes from
//! hooks and append the resulting documents to a turn's extra context.
//!
//! Indexes can also serve custom architectures that use multiple LLMs or agents.
//!
//! ## Conversation memory
//! Runtimes can load and persist per-conversation history through the
//! [ConversationMemory](crate::memory::ConversationMemory) trait. The classic
//! `rig-agent` runtime integrates this portable backend contract.
//! The default in-process backend
//! [InMemoryConversationMemory](crate::memory::InMemoryConversationMemory) is suitable
//! for tests and single-process agents; reusable history-shaping policies (sliding
//! window, token budget) live in the [`rig-memory`](https://crates.io/crates/rig-memory)
//! companion crate. See [`examples/agent_with_memory.rs`](https://github.com/0xPlaygrounds/rig/blob/main/examples/agent_with_memory.rs)
//! for a runnable end-to-end example.
//!
//! # Integrations
//! ## Model Providers
//! Rig natively supports the following completion and embedding model provider integrations:
//! - Anthropic
//! - Azure OpenAI
//! - ChatGPT and GitHub Copilot auth-backed clients
//! - Cohere
//! - DeepSeek
//! - Gemini
//! - Groq
//! - Hugging Face
//! - Hyperbolic
//! - Llamafile
//! - MiniMax
//! - Mira
//! - Mistral
//! - Moonshot
//! - Ollama
//! - OpenAI
//! - OpenRouter
//! - Perplexity
//! - Together
//! - Voyage AI
//! - xAI
//! - Xiaomi MiMo
//! - Z.ai
//!
//! You can also implement your own model provider integration by defining types that
//! implement the [CompletionModel](crate::completion::CompletionModel) and [EmbeddingModel](crate::embeddings::EmbeddingModel) traits.
//!
//! Vector stores are available as separate companion-crates:
//!
//! - MongoDB: [`rig-mongodb`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-mongodb)
//! - LanceDB: [`rig-lancedb`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-lancedb)
//! - Neo4j: [`rig-neo4j`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-neo4j)
//! - Qdrant: [`rig-qdrant`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-qdrant)
//! - SQLite: [`rig-sqlite`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-sqlite)
//! - SurrealDB: [`rig-surrealdb`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-surrealdb)
//! - Milvus: [`rig-milvus`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-milvus)
//! - ScyllaDB: [`rig-scylladb`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-scylladb)
//! - AWS S3Vectors: [`rig-s3vectors`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-s3vectors)
//! - HelixDB: [`rig-helixdb`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-helixdb)
//! - Cloudflare Vectorize: [`rig-vectorize`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-vectorize)
//!
//! You can also implement your own vector store integration by defining types that
//! implement the [VectorStoreIndex](crate::vector_store::VectorStoreIndex) trait.
//!
//! The following providers are available as separate companion-crates:
//!
//! - AWS Bedrock: [`rig-bedrock`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-bedrock)
//! - Fastembed: [`rig-fastembed`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-fastembed)
//! - Google Gemini gRPC: [`rig-gemini-grpc`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-gemini-grpc)
//! - Google Vertex AI: [`rig-vertexai`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-vertexai)
//!

extern crate self as rig;

#[cfg(feature = "audio")]
#[cfg_attr(docsrs, doc(cfg(feature = "audio")))]
pub mod audio_generation;
pub mod client;
pub mod completion;
pub mod embeddings;
pub mod http_client;
pub mod id;
#[cfg(feature = "image")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
pub mod image_generation;
/// Internal JSON helpers shared with sibling runtime crates (e.g. `rig-agent`).
/// Not part of rig-core's stable public API.
#[doc(hidden)]
pub mod json_utils;
pub mod loaders;
pub mod markers;
pub mod memory;
pub mod model;
pub mod one_or_many;
pub mod prelude;
pub(crate) mod provider_response;
pub mod providers;
pub mod rerank;

pub mod streaming;
#[cfg(any(test, feature = "test-utils"))]
#[cfg_attr(docsrs, doc(cfg(feature = "test-utils")))]
pub mod test_utils;
pub mod tool;
pub mod transcription;
pub mod vector_store;
pub mod wasm_compat;

// Re-export commonly used types and traits
pub use completion::message;
pub use embeddings::Embed;
pub use one_or_many::{EmptyListError, OneOrMany};
pub use provider_response::ProviderResponseError;
// `schemars`, `serde`, and `serde_json` are re-exported so macro-generated
// code (and downstream crates) can resolve them through Rig instead of
// requiring a direct dependency on each.
pub use schemars;
pub use serde;
pub use serde_json;

#[cfg(feature = "derive")]
#[cfg_attr(docsrs, doc(cfg(feature = "derive")))]
pub use rig_derive::Embed;

// The portable `#[rig_tool]` macro produces context-free `PortableTool`s, which
// are rig-core-owned, so direct `rig-core` dependents can reach it without
// pulling in `rig-derive` themselves.
#[cfg(feature = "derive")]
#[cfg_attr(docsrs, doc(cfg(feature = "derive")))]
pub use rig_derive::{rig_tool, rig_tool as tool_macro};

pub mod telemetry;
