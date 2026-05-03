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
//! ```ignore
//! use rig_core::{
//!     client::{CompletionClient, ProviderClient},
//!     completion::Prompt,
//!     providers::openai,
//! };
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create OpenAI client and agent.
//!     // This requires the `OPENAI_API_KEY` environment variable to be set.
//!     let openai_client = openai::Client::from_env()?;
//!
//!     let agent = openai_client.agent(openai::GPT_5_2).build();
//!
//!     // Prompt the model and print its response
//!     let response = agent
//!         .prompt("Who are you?")
//!         .await?;
//!
//!     println!("{response}");
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
//! ## Agents
//! Rig also provides high-level abstractions over LLMs in the form of the [Agent](crate::agent::Agent) type.
//!
//! The [Agent](crate::agent::Agent) type can be used to create anything from simple agents that use vanilla models to full blown
//! RAG systems that can be used to answer questions using a knowledge base.
//!
//! ## Vector stores and indexes
//! Rig provides a common interface for working with vector stores and indexes. Specifically, the library
//! provides the [VectorStoreIndex](crate::vector_store::VectorStoreIndex)
//! trait, which can be implemented to define vector stores and indices respectively.
//! Those can then be used as the knowledge base for a RAG enabled [Agent](crate::agent::Agent), or
//! as a source of context documents in a custom architecture that use multiple LLMs or agents.
//!
//! # Integrations
//! ## Model Providers
//! Rig natively supports the following completion and embedding model provider integrations:
//! - Anthropic
//! - Azure
//! - Cohere
//! - Deepseek
//! - Galadriel
//! - Gemini
//! - Groq
//! - Huggingface
//! - Hyperbolic
//! - Mira
//! - Mistral
//! - Moonshot
//! - Ollama
//! - Openai
//! - OpenRouter
//! - Perplexity
//! - Together
//! - Voyage AI
//! - xAI
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

pub mod agent;
#[cfg(feature = "audio")]
#[cfg_attr(docsrs, doc(cfg(feature = "audio")))]
pub mod audio_generation;
pub mod client;
pub mod completion;
pub mod embeddings;

#[cfg(feature = "experimental")]
#[cfg_attr(docsrs, doc(cfg(feature = "experimental")))]
pub mod evals;
pub mod extractor;
pub mod http_client;
#[cfg(feature = "image")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
pub mod image_generation;
pub mod integrations;
pub(crate) mod json_utils;
pub mod loaders;
pub mod markers;
pub mod model;
pub mod one_or_many;
pub mod pipeline;
pub mod prelude;
pub mod providers;

pub mod streaming;
pub mod tool;
pub mod tools;
pub mod transcription;
pub mod vector_store;
pub mod wasm_compat;

// Re-export commonly used types and traits
pub use completion::message;
pub use embeddings::Embed;
pub use extractor::ExtractionResponse;
pub use one_or_many::{EmptyListError, OneOrMany};

#[cfg(feature = "derive")]
#[cfg_attr(docsrs, doc(cfg(feature = "derive")))]
pub use rig_derive::{Embed, rig_tool as tool_macro};

pub mod telemetry;
