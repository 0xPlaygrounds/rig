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
//! - Simple but powerful common abstractions over LLM providers (e.g. OpenAI, Cohere), embeddings, tools, and hooks
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
//! ## Retrieval (RAG)
//! Rig does not ship a built-in vector-store abstraction. Retrieval is a user-land
//! pattern: expose it as a normal [tool](crate::tool) the model chooses to call, or
//! inject retrieved context before each model call from an
//! [AgentHook](crate::agent::AgentHook) via
//! [RequestPatch::extra_context](crate::agent::RequestPatch). Embedding
//! models/builders remain in [embeddings](crate::embeddings). For catalogs with more
//! tools than fit in a prompt, register the extras as deferred tools and let the model
//! discover them with the built-in `tool_search` meta-tool
//! (see [`ToolServer::deferred_tool`](crate::tool::ToolServer::deferred_tool)).
//!
//! ## Conversation memory
//! Rig can transparently load and persist per-conversation history through the
//! [ConversationMemory](crate::memory::ConversationMemory) trait. Attach a backend
//! with [`AgentBuilder::memory`](crate::agent::AgentBuilder::memory) and identify the
//! conversation per-request via
//! [`PromptRequest::conversation`](crate::agent::prompt_request::PromptRequest::conversation).
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
//! - Galadriel
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
pub mod id;
#[cfg(feature = "image")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
pub mod image_generation;
pub mod integrations;
pub(crate) mod json_utils;
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
pub mod tools;
pub mod transcription;
pub mod wasm_compat;

// Re-export commonly used types and traits
pub use completion::message;
pub use embeddings::Embed;
pub use extractor::ExtractionResponse;
pub use one_or_many::{EmptyListError, OneOrMany};
pub use provider_response::ProviderResponseError;
pub use schemars;

#[cfg(feature = "derive")]
#[cfg_attr(docsrs, doc(cfg(feature = "derive")))]
pub use rig_derive::{Embed, rig_tool as tool_macro};

pub mod telemetry;
