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
//! ```no_run
//! use rig::{
//!     client::{CompletionClient, ProviderClient},
//!     completion::{Chat, Completion, Prompt},
//!     providers::openai,
//! };
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let openai = openai::Client::from_env()?;
//!
//! // Configure the agent
//! let agent = openai.agent(openai::GPT_5_2)
//!     .preamble("System prompt")
//!     .context("Context document 1")
//!     .context("Context document 2")
//!     .temperature(0.8)
//!     .build();
//!
//! // Use the agent for completions and prompts
//! // Generate a chat completion response from a prompt and chat history
//! let chat_response = agent.chat("Prompt", Vec::<rig::completion::Message>::new()).await?;
//!
//! // Generate a prompt completion response from a simple prompt
//! let prompt_response = agent.prompt("Prompt").await?;
//!
//! // Generate a completion request builder from a prompt and chat history. The builder
//! // will contain the agent's configuration (i.e.: preamble, context documents, tools,
//! // model parameters, etc.), but these can be overwritten.
//! let completion_req_builder = agent
//!     .completion("Prompt", Vec::<rig::completion::Message>::new())
//!     .await?;
//!
//! let response = completion_req_builder
//!     .temperature(0.9) // Overwrite the agent's temperature
//!     .send()
//!     .await?;
//! # Ok(())
//! # }
//! ```
//!
//! RAG Agent example
//! ```no_run
//! use rig::{
//!     client::{CompletionClient, EmbeddingsClient, ProviderClient},
//!     completion::Prompt,
//!     embeddings::EmbeddingsBuilder,
//!     providers::openai,
//!     vector_store::in_memory_store::InMemoryVectorStore,
//! };
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! // Initialize OpenAI client
//! let openai = openai::Client::from_env()?;
//!
//! // Initialize OpenAI embedding model
//! let embedding_model = openai.embedding_model(openai::TEXT_EMBEDDING_3_SMALL);
//!
//! // Create vector store, compute embeddings and load them in the store
//! let mut vector_store = InMemoryVectorStore::default();
//!
//! let embeddings = EmbeddingsBuilder::new(embedding_model.clone())
//!     .documents(vec![
//!         "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets",
//!         "Definition of a *glarb-glarb*: A glarb-glarb is an ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.",
//!         "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.",
//!     ])?
//!     .build()
//!     .await?;
//!
//! vector_store.add_documents(embeddings);
//!
//! // Create vector store index
//! let index = vector_store.index(embedding_model);
//!
//! let agent = openai.agent(openai::GPT_5_2)
//!     .preamble("
//!         You are a dictionary assistant here to assist the user in understanding the meaning of words.
//!         You will find additional non-standard word definitions that could be useful below.
//!     ")
//!     .dynamic_context(1, index)
//!     .build();
//!
//! // Prompt the agent and print the response
//! let response = agent.prompt("What does \"glarb-glarb\" mean?").await?;
//! # Ok(())
//! # }
//! ```
mod builder;
mod completion;
pub(crate) mod prompt_request;
mod tool;

pub use crate::message::Text;
pub use builder::{AgentBuilder, NoToolConfig, WithBuilderTools, WithToolServerHandle};
pub use completion::Agent;
pub use prompt_request::hooks::{HookAction, PromptHook, ToolCallHookAction};
pub use prompt_request::streaming::{
    FinalResponse, MultiTurnStreamItem, StreamingError, StreamingPromptRequest, StreamingResult,
    stream_to_stdout,
};
pub use prompt_request::{PromptRequest, PromptResponse, TypedPromptRequest, TypedPromptResponse};
