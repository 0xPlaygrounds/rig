//! This module contains the implementation of the [Agent] struct and its builder.
//!
//! The [Agent] struct represents an LLM agent, which combines an LLM model with a preamble (system prompt),
//! a set of context documents, and a set of tools. Note: both context documents and tools can be either
//! static (i.e.: they are always provided) or dynamic (i.e.: they are RAGged at prompt-time).
//!
//! The [Agent] struct is highly configurable, allowing the user to define anything from
//! a simple bot with a specific system prompt to a complex RAG system with a set of dynamic
//! context documents and tools.
//!
//! The [Agent] struct implements the [crate::completion::Completion] and [crate::completion::Prompt] traits,
//! allowing it to be used for generating completions responses and prompts. The [Agent] struct also
//! implements the [crate::completion::Chat] trait, which allows it to be used for generating chat completions.
//!
//! The [AgentBuilder] implements the builder pattern for creating instances of [Agent].
//! It allows configuring the model, preamble, context documents, tools, temperature, and additional parameters
//! before building the agent.
//!
//! # Example
//! ```rust
//! use rig::{
//!     completion::{Chat, Completion, Prompt},
//!     providers::openai,
//! };
//!
//! let openai = openai::Client::from_env();
//!
//! // Configure the agent
//! let agent = openai.agent("gpt-4o")
//!     .preamble("System prompt")
//!     .context("Context document 1")
//!     .context("Context document 2")
//!     .tool(tool1)
//!     .tool(tool2)
//!     .temperature(0.8)
//!     .additional_params(json!({"foo": "bar"}))
//!     .build();
//!
//! // Use the agent for completions and prompts
//! // Generate a chat completion response from a prompt and chat history
//! let chat_response = agent.chat("Prompt", chat_history)
//!     .await
//!     .expect("Failed to chat with Agent");
//!
//! // Generate a prompt completion response from a simple prompt
//! let chat_response = agent.prompt("Prompt")
//!     .await
//!     .expect("Failed to prompt the Agent");
//!
//! // Generate a completion request builder from a prompt and chat history. The builder
//! // will contain the agent's configuration (i.e.: preamble, context documents, tools,
//! // model parameters, etc.), but these can be overwritten.
//! let completion_req_builder = agent.completion("Prompt", chat_history)
//!     .await
//!     .expect("Failed to create completion request builder");
//!
//! let response = completion_req_builder
//!     .temperature(0.9) // Overwrite the agent's temperature
//!     .send()
//!     .await
//!     .expect("Failed to send completion request");
//! ```
//!
//! RAG Agent example
//! ```rust
//! use rig::{
//!     completion::Prompt,
//!     embeddings::EmbeddingsBuilder,
//!     providers::openai,
//!     vector_store::{in_memory_store::InMemoryVectorStore, VectorStore},
//! };
//!
//! // Initialize OpenAI client
//! let openai = openai::Client::from_env();
//!
//! // Initialize OpenAI embedding model
//! let embedding_model = openai.embedding_model(openai::TEXT_EMBEDDING_ADA_002);
//!
//! // Create vector store, compute embeddings and load them in the store
//! let mut vector_store = InMemoryVectorStore::default();
//!
//! let embeddings = EmbeddingsBuilder::new(embedding_model.clone())
//!     .simple_document("doc0", "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets")
//!     .simple_document("doc1", "Definition of a *glarb-glarb*: A glarb-glarb is a ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.")
//!     .simple_document("doc2", "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.")
//!     .build()
//!     .await
//!     .expect("Failed to build embeddings");
//!
//! vector_store.add_documents(embeddings)
//!     .await
//!     .expect("Failed to add documents");
//!
//! // Create vector store index
//! let index = vector_store.index(embedding_model);
//!
//! let agent = openai.agent(openai::GPT_4O)
//!     .preamble("
//!         You are a dictionary assistant here to assist the user in understanding the meaning of words.
//!         You will find additional non-standard word definitions that could be useful below.
//!     ")
//!     .dynamic_context(1, index)
//!     .build();
//!
//! // Prompt the agent and print the response
//! let response = agent.prompt("What does \"glarb-glarb\" mean?").await
//!     .expect("Failed to prompt the agent");
//! ```
mod builder;
mod completion;
pub(crate) mod prompt_request;
mod tool;

pub use crate::message::Text;
pub use builder::{AgentBuilder, AgentBuilderSimple};
pub use completion::Agent;
pub use prompt_request::PromptHook;
pub use prompt_request::streaming::{
    FinalResponse, MultiTurnStreamItem, StreamingPromptRequest, stream_to_stdout,
};
pub use prompt_request::{CancelSignal, PromptRequest, PromptResponse};
