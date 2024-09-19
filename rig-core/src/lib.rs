//! Rig is a Rust library for building LLM-powered applications that focuses on ergonomics and modularity.
//!
//! # Table of contents
//!
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
//!     // Create OpenAI client and model.
//!     // This requires the `OPENAI_API_KEY` environment variable to be set.
//!     let openai_client = openai::Client::from_env();
//!
//!     let gpt4 = openai_client.model("gpt-4").build();
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
//! each provider (e.g. OpenAI, Cohere) has a `Client` struct that can be used to create completion
//! and embedding models. These models implement the [CompletionModel](crate::completion::CompletionModel)
//! and [EmbeddingModel](crate::embeddings::EmbeddingModel) traits respectively, which provide a common,
//! low-level interface for creating completion and embedding requests and executing them.
//!
//! ## Models, Agents and RagAgents
//! Rig provides high-level abstractions over LLMs in the form of [Model](crate::model::Model),
//! [Agent](crate::agent::Agent) and [RagAgent](crate::rag::RagAgent) structs.
//!
//! These structs range from simple models that can be prompted directly to agents that have a
//! system prompt to full blown RAG systems that can be used to answer questions using a knowledgebase.
//! Here is a quick summary of each:
//! - [Model](crate::model::Model): A simple LLM model that can be prompted directly. This structs acts
//!   as a thin wrapper around a completion model (i.e.: a struct implementing the [CompletionModel](crate::completion::CompletionModel) trait).
//! - [Agent](crate::agent::Agent): An LLM model combined with a preamble (i.e.: system prompt) and a
//!   static set of context documents and tools.
//! - [RagAgent](crate::rag::RagAgent): A RAG system that can be used to answer questions using a knowledgebase
//!   containing both context documents and tools.
//!
//! ## Vector stores and indexes
//! Rig provides a common interface for working with vector stores and indexes. Specifically, the library
//! provides the [VectorStore](crate::vector_store::VectorStore) and [VectorStoreIndex](crate::vector_store::VectorStoreIndex)
//! traits, which can be implemented to define vector stores and indices respectively.
//! Those can then be used as the knowledgebase for a [RagAgent](crate::rag::RagAgent), or
//! as a source of context documents in a custom architecture that use multiple LLMs or agents.
//!
//! # Integrations
//! Rig natively supports the following completion and embedding model providers:
//! - OpenAI
//! - Cohere
//!
//! Rig currently has the following integration companion crates:
//! - `rig-mongodb`: Vector store implementation for MongoDB
//!

pub mod agent;
pub mod cli_chatbot;
pub mod completion;
pub mod document_loaders;
pub mod embeddings;
pub mod extractor;
pub mod json_utils;
pub mod model;
pub mod providers;
pub mod rag;
pub mod tool;
pub mod vector_store;
