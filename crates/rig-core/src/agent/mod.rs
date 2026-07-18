//! This module contains the implementation of the [Agent] struct and its builder.
//!
//! The [Agent] struct represents an LLM agent, which combines an LLM model with a preamble (system prompt),
//! a set of static context documents, and a set of tools. Tools can be always
//! available or selected from a retrieval index at prompt time.
//!
//! The [Agent] struct is highly configurable, allowing the user to define anything from
//! a simple bot with a specific system prompt to a complex RAG system.
//!
//! The [Agent] struct implements the runner-backed [crate::completion::Prompt],
//! [crate::completion::TypedPrompt], and [crate::completion::Chat] traits. All
//! agent execution goes through [AgentRunner], so hooks and lifecycle policies
//! cannot be bypassed through a raw agent request builder.
//!
//! The [AgentBuilder] implements the builder pattern for creating instances of [Agent].
//! It allows configuring the model, preamble, context documents, tools, temperature, and additional parameters
//! before building the agent.
//!
//! # Example
//! ```no_run
//! use rig_core::{
//!     client::{CompletionClient, ProviderClient},
//!     completion::{Chat, Prompt},
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
//! Passive RAG is application-defined: a completion-call hook chooses the query,
//! retrieves documents, and injects them with [`RequestPatch::extra_context`].
//! Active RAG instead exposes a vector index or custom retriever as a tool so the
//! model decides when and how to search.
//!
//! Passive RAG agent example
//! ```no_run
//! use rig_core::{
//!     agent::{AgentHook, CompletionCallAction, CompletionCallEvent, HookContext, RequestPatch},
//!     client::{CompletionClient, EmbeddingsClient, ProviderClient},
//!     completion::{Document, Message, Prompt},
//!     embeddings::EmbeddingsBuilder,
//!     message::UserContent,
//!     providers::openai,
//!     vector_store::{
//!         VectorStoreIndexDyn,
//!         in_memory_store::InMemoryVectorStore,
//!         request::VectorSearchRequest,
//!     },
//! };
//!
//! struct DictionaryRag<I>(I);
//!
//! impl<I> AgentHook for DictionaryRag<I>
//! where
//!     I: VectorStoreIndexDyn,
//! {
//!     async fn on_completion_call(
//!         &self,
//!         _ctx: &HookContext,
//!         event: CompletionCallEvent<'_>,
//!     ) -> CompletionCallAction {
//!         let message_text = |message: &Message| match message {
//!             Message::User { content } => content.iter().find_map(|item| match item {
//!                 UserContent::Text(text) => Some(text.text.clone()),
//!                 _ => None,
//!             }),
//!             _ => None,
//!         };
//!         let Some(query) = message_text(event.prompt)
//!             .or_else(|| event.history.iter().rev().find_map(message_text))
//!         else {
//!             return CompletionCallAction::continue_run();
//!         };
//!
//!         let request = VectorSearchRequest::builder().query(query).samples(1).build();
//!         match self.0.top_n(request).await {
//!             Ok(results) => CompletionCallAction::patch(RequestPatch::new().extra_context(
//!                 results.into_iter().map(|(_, id, value)| Document {
//!                     id,
//!                     text: serde_json::to_string_pretty(&value)
//!                         .unwrap_or_else(|_| value.to_string()),
//!                     additional_props: Default::default(),
//!                 }),
//!             )),
//!             Err(error) => CompletionCallAction::stop(format!("retrieval failed: {error}")),
//!         }
//!     }
//! }
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
//!     .add_hook(DictionaryRag(index))
//!     .build();
//!
//! // Prompt the agent and print the response
//! let response = agent.prompt("What does \"glarb-glarb\" mean?").await?;
//! # Ok(())
//! # }
//! ```
mod builder;
mod completion;
pub mod hook;
pub(crate) mod prompt_request;
pub mod run;
pub mod runner;
mod tool;

/// Fallback display name used in telemetry spans and logs when an agent has no
/// configured name.
pub(crate) const UNKNOWN_AGENT_NAME: &str = "Unnamed Agent";

pub use crate::message::Text;
pub use builder::{AgentBuilder, NoToolConfig, WithBuilderTools, WithToolServerHandle};
pub use completion::Agent;
pub use hook::CompletionCall as CompletionCallEvent;
pub use hook::{
    AgentHook, CompletionCallAction, CompletionResponse as CompletionResponseEvent, HookContext,
    HookStack, InvalidToolCallAction, InvalidToolCallContext, ModelTurnAction, ModelTurnFinished,
    ObservationAction, RequestPatch, RetryRequest, RunId, Scratchpad, StepEventKind,
    StreamResponseFinish, TextDelta, ToolCall, ToolCallAction, ToolCallDelta, ToolResultAction,
    ToolResultEvent,
};
pub use prompt_request::streaming::{
    MultiTurnStreamItem, StreamingError, StreamingPromptRequest, StreamingResult, stream_to_stdout,
};
pub use prompt_request::{
    CompletionCall, PromptRequest, PromptResponse, TypedPromptRequest, TypedPromptResponse,
};
pub use run::{AgentRun, AgentRunStep, ModelTurn, ModelTurnOutcome, OutputMode, PendingToolCall};
pub use runner::AgentRunner;
