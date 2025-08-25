#![cfg_attr(docsrs, feature(doc_cfg))]
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
//! # Simple example:
//! ```
//! use rig::{completion::Prompt, providers::openai};
//!
//! #[tokio::main]
//! async fn main() {
//!     // Create OpenAI client and agent.
//!     // This requires the `OPENAI_API_KEY` environment variable to be set.
//!     let openai_client = openai::Client::from_env();
//!
//!     let gpt4 = openai_client.agent("gpt-4").build();
//!
//!     // Prompt the model and print its response
//!     let response = gpt4
//!         .prompt("Who are you?")
//!         .await
//!         .expect("Failed to prompt GPT-4");
//!
//!     println!("GPT-4: {response}");
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
//! - OpenAI
//! - Cohere
//! - Anthropic
//! - Perplexity
//! - Google Gemini
//! - xAI
//! - DeepSeek
//!
//! You can also implement your own model provider integration by defining types that
//! implement the [CompletionModel](crate::completion::CompletionModel) and [EmbeddingModel](crate::embeddings::EmbeddingModel) traits.
//!
//! ## Vector Stores
//! Rig currently supports the following vector store integrations via companion crates:
//! - `rig-mongodb`: Vector store implementation for MongoDB
//! - `rig-lancedb`: Vector store implementation for LanceDB
//! - `rig-neo4j`: Vector store implementation for Neo4j
//! - `rig-qdrant`: Vector store implementation for Qdrant
//!
//! You can also implement your own vector store integration by defining types that
//! implement the [VectorStoreIndex](crate::vector_store::VectorStoreIndex) trait.

extern crate self as rig;

pub mod agent;
#[cfg(feature = "audio")]
pub mod audio_generation;
pub mod cli_chatbot;
pub mod client;
pub mod completion;
pub mod embeddings;
pub mod extractor;
#[cfg(feature = "image")]
pub mod image_generation;
pub(crate) mod json_utils;
pub mod loaders;
pub mod one_or_many;
pub mod pipeline;
pub mod prelude;
pub mod providers;
pub mod streaming;
pub mod think_tool;
pub mod tool;
pub mod transcription;
pub mod vector_store;

// Re-export commonly used types and traits
pub use completion::message;
pub use embeddings::Embed;
pub use one_or_many::{EmptyListError, OneOrMany};

#[cfg(feature = "derive")]
pub use rig_derive::Embed;
