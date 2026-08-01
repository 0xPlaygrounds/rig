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
//!     completion::AssistantContent,
//!     http_runtime::HttpRuntime,
//!     providers::openai,
//! };
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Configure an OpenAI completion call.
//!     // This requires the `OPENAI_API_KEY` environment variable to be set.
//!     let cfg = openai::functions::Config::from_env(openai::GPT_5_2)?;
//!     let rt = HttpRuntime::new();
//!
//!     let request = rig_core::completion::CompletionRequest::from_prompt("Who are you?");
//!     let response = openai::functions::complete(&cfg, &rt, request).await?;
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
//! ## Completion and embedding data
//! Rig provides a consistent API for working with LLMs and embeddings. Specifically,
//! each provider (e.g. OpenAI, Cohere) has a `functions` module holding a plain
//! `Config` (model id, credentials, base URL, provider-specific knobs) plus free
//! functions — `complete`, `stream`, `embed`, `transcribe`, and so on — that take
//! `(&Config, &HttpRuntime, request)` and return the shared
//! [CompletionResponse](crate::completion::CompletionResponse),
//! [Embedding](crate::embeddings::Embedding), and sibling data types. The
//! [HttpRuntime](crate::http_runtime::HttpRuntime) owns transport concerns and is
//! shared across providers; static provider facts are described by
//! [ProviderDescriptor](crate::providers::ProviderDescriptor).
//!
//! ## Agent runtimes
//! This crate owns the provider-agnostic model, message, tool, and storage
//! contracts. The sibling `rig-agent` crate provides the classic builder and
//! run-loop API.
//!
//! ## Vector stores and indexes
//! Rig provides a common data vocabulary for working with vector stores:
//! [VectorSearchRequest](crate::vector_store::VectorSearchRequest) (pre-embedded queries),
//! [SearchHit](crate::vector_store::SearchHit), and
//! [StoreRecord](crate::vector_store::StoreRecord). Store crates expose concrete
//! inherent async methods (`top_n`, `top_n_ids`, `top_n_as`, `insert`, `insert_as`)
//! over these types. Stores can be queried directly by applications or runtimes.
//! For active RAG, expose a store through a custom tool so the model decides when
//! and how to retrieve. The classic `rig-agent` runtime can also query stores from
//! hooks and append the resulting documents to a turn's extra context.
//!
//! Stores can also serve custom architectures that use multiple LLMs or agents.
//!
//! ## Conversation memory
//! Conversation history is host-owned data: nothing in Rig loads or saves it
//! behind your back. The in-process store
//! [InMemoryConversationMemory](crate::memory::InMemoryConversationMemory) has
//! plain `load`/`append`/`clear` methods and is suitable for tests and
//! single-process agents; hosts with a database implement whatever store they
//! like and report failures as
//! [MemoryError](crate::memory::MemoryError). Reusable history-shaping
//! policies (sliding window, token budget, rolling summaries) live in the
//! [`rig-memory`](https://crates.io/crates/rig-memory) companion crate as
//! plain data. See [`examples/agent_with_memory.rs`](https://github.com/0xPlaygrounds/rig/blob/main/examples/agent_with_memory.rs)
//! for the runnable load-before / append-after recipe.
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
//! You can also add your own model provider integration by writing a `Config` plus
//! free functions that build `http::Request`s, send them through
//! [HttpRuntime](crate::http_runtime::HttpRuntime), and parse the responses into
//! Rig's shared request/response data types.
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
//! You can also implement your own vector store integration by exposing the same
//! inherent methods over the shared vector store data types.
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
pub mod completion;
pub mod embeddings;
pub mod http_client;
pub mod http_runtime;
pub mod id;
#[cfg(feature = "image")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
pub mod image_generation;
/// Internal JSON helpers shared with sibling runtime crates (e.g. `rig-agent`).
/// Not part of rig-core's stable public API.
#[doc(hidden)]
pub mod json_utils;
pub mod loaders;
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
