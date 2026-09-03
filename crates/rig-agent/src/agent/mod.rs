//! This module contains the implementation of the [Agent] struct and its builder.
//!
//! The [Agent] struct represents an LLM agent, which combines an LLM model with a preamble (system prompt),
//! a set of static context documents, and a set of tools. Tools can be always
//! available or selected from a retrieval index at prompt time.
//!
//! The [Agent] struct is highly configurable, allowing the user to define anything from
//! a simple bot with a specific system prompt to a complex RAG system.
//!
//! The [Agent] struct exposes the runner-backed [Agent::prompt],
//! [Agent::prompt_typed], and [Agent::chat] methods. All
//! agent execution goes through [AgentRunner], so hooks and lifecycle policies
//! cannot be bypassed through a raw agent request builder.
//!
//! The [AgentBuilder] implements the builder pattern for creating instances of [Agent].
//! It allows configuring the model, preamble, context documents, tools, temperature, and additional parameters
//! before building the agent.
//!
//! # Example
//! ```no_run
//! use rig_agent::prelude::*;
//! use rig_core::providers::openai;
//! use rig_reqwest::prelude::*;
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
//! // Use the agent for chats and prompts
//! // Generate a chat completion response from a prompt and chat history
//! let chat_response = agent.chat("Prompt", &mut Vec::<rig_core::completion::Message>::new()).await?;
//!
//! // Generate a prompt completion response from a simple prompt
//! let prompt_response = agent.prompt("Prompt").await?;
//!
//! // Per-run overrides stay inside the hook-aware runner.
//! let response = agent.runner("Prompt").temperature(0.9).run().await?;
//! # Ok(())
//! # }
//! ```
//!
//! [`AgentBuilder::dynamic_context`] provides passive RAG through the same
//! completion-call hook lifecycle as every other request policy. For custom
//! query selection, filtering, reranking, caching, formatting, or failure
//! handling, applications can instead implement [`AgentHook`] and inject
//! documents with [`RequestPatch::extra_context`]. Active RAG exposes a vector
//! index or custom retriever as a tool so the model decides when to search.
//!
//! Passive RAG agent example
//! ```no_run
//! use rig_agent::prelude::*;
//! use rig_reqwest::prelude::*;
//! use rig_core::{
//!     client::EmbeddingsClient,
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
pub(crate) mod bus;
mod completion;
mod engine;
pub mod hook;
pub mod run;
pub mod runner;
mod streaming;
mod telemetry;
mod tool;
mod typed;

/// Fallback display name used in telemetry spans and logs when an agent has no
/// configured name.
pub(crate) const UNKNOWN_AGENT_NAME: &str = "Unnamed Agent";

pub use crate::run::response::{CompletionCall, PromptResponse};
pub use crate::run::spec::RunSpec;
pub use builder::{AgentBuilder, NoToolConfig, WithBuilderTools, WithToolServerHandle};
pub use completion::{Agent, AgentParts};
pub use hook::CompletionCall as CompletionCallEvent;
pub use hook::{
    AgentHook, CompletionCallAction, HookContext, HookStack, InvalidToolCallAction,
    InvalidToolCallContext, ModelSelection, ModelSelectionAction, ModelTurnAction,
    ModelTurnFinished, ObservationAction, ReasoningDelta, RequestPatch, RetryRequest, RunEntry,
    RunId, RunSettled, RunStart, RunStartAction, Scratchpad, SettledOutcome, StepEventKind,
    TextDelta, ToolCallDelta,
};
pub use hook::{DispatchAction, DispatchEvent, OutcomeAction, OutcomeEvent};
pub use rig_core::bus::ModelHandle;
pub use rig_core::completion::ModelRef;
/// The provider-neutral identity carrier, re-exported from rig-core so agent
/// callers name one type across core responses, stream terminals, completion
/// calls, and hook events.
pub use rig_core::completion::ResponseIdentity;
pub use rig_core::message::Text;
pub use run::TurnTools;
pub use run::{AgentRun, AgentRunStep, ModelTurn, ModelTurnOutcome, OutputMode, PendingToolCall};
pub use runner::AgentRunner;
pub use streaming::{
    MultiTurnStreamItem, RUN_EVENTS_CAPACITY, RunEvents, StreamingError, StreamingResult,
    stream_to_stdout,
};
pub use typed::{TypedPromptResponse, TypedRun};
