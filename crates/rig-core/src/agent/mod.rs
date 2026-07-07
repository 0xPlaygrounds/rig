//! This module contains the implementation of the [Agent] struct and its builder.
//!
//! The [Agent] struct represents an LLM agent, which combines an LLM model with a preamble (system prompt),
//! a set of context documents, and a set of tools. Context documents and tools are registered when the
//! agent is built. Rig ships no built-in vector store, so retrieval (RAG) is a user-land pattern: inject
//! per-turn context from an [`AgentHook`](crate::agent::AgentHook) via
//! [`RequestPatch::extra_context`](crate::agent::RequestPatch), or expose retrieval as a
//! [`Tool`](crate::tool::Tool) the model calls (see the RAG example below).
//!
//! The [Agent] struct is highly configurable, allowing the user to define anything from
//! a simple bot with a specific system prompt to a full RAG system built from hooks and tools.
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
//! use rig_core::{
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
//! let chat_response = agent.chat("Prompt", &mut Vec::<rig_core::completion::Message>::new()).await?;
//!
//! // Generate a prompt completion response from a simple prompt
//! let prompt_response = agent.prompt("Prompt").await?;
//!
//! // Generate a completion request builder from a prompt and chat history. The builder
//! // will contain the agent's configuration (i.e.: preamble, context documents, tools,
//! // model parameters, etc.), but these can be overwritten.
//! let completion_req_builder = agent
//!     .completion("Prompt", Vec::<rig_core::completion::Message>::new())
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
//! RAG agent example — retrieval is a user-land pattern. Rig ships no built-in
//! vector store: inject retrieved context before each model call from an
//! [`AgentHook`](crate::agent::AgentHook) via
//! [`RequestPatch::extra_context`](crate::agent::RequestPatch) (passive RAG, shown
//! below), or expose retrieval as an ordinary [`Tool`](crate::tool::Tool) the model
//! calls (active RAG). See the `hook_passive_rag` and `tool_active_rag` examples
//! for runnable end-to-end versions.
//! ```no_run
//! use std::collections::HashMap;
//!
//! use rig_core::agent::{AgentHook, Flow, HookContext, RequestPatch, StepEvent};
//! use rig_core::client::{CompletionClient, ProviderClient};
//! use rig_core::completion::{CompletionModel, Document, Message, Prompt};
//! use rig_core::message::UserContent;
//! use rig_core::providers::openai;
//!
//! // A tiny in-process knowledge base — swap for embeddings + your own store.
//! const KB: &[(&str, &str)] = &[
//!     ("glarb-glarb", "A glarb-glarb is an ancient tool used to farm the land."),
//!     ("flurbo", "A flurbo is a green alien that lives on cold planets."),
//! ];
//!
//! // Injects retrieved context on the first model call, before the provider sees it.
//! struct DictionaryRag;
//!
//! impl<M: CompletionModel> AgentHook<M> for DictionaryRag {
//!     async fn on_event(&self, _ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
//!         if let StepEvent::CompletionCall { prompt, turn, .. } = event
//!             && turn == 1
//!         {
//!             // Read the query from the prompt (public Message/UserContent API).
//!             let Message::User { content } = prompt else {
//!                 return Flow::cont();
//!             };
//!             let Some(query) = content.iter().find_map(|c| match c {
//!                 UserContent::Text(t) => Some(t.text()),
//!                 _ => None,
//!             }) else {
//!                 return Flow::cont();
//!             };
//!             // Lowercase and strip surrounding punctuation so a query token
//!             // like `"glarb-glarb"` or `mean?` still matches the bare word.
//!             let query = query.to_lowercase();
//!             let docs: Vec<Document> = KB
//!                 .iter()
//!                 .filter(|(_, text)| {
//!                     let text = text.to_lowercase();
//!                     query
//!                         .split_whitespace()
//!                         .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
//!                         .any(|w| !w.is_empty() && text.contains(w))
//!                 })
//!                 .map(|(id, text)| Document {
//!                     id: (*id).to_string(),
//!                     text: (*text).to_string(),
//!                     additional_props: HashMap::new(),
//!                 })
//!                 .collect();
//!             return Flow::patch_request(RequestPatch::new().extra_context(docs));
//!         }
//!         Flow::cont()
//!     }
//! }
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let openai = openai::Client::from_env()?;
//! let agent = openai
//!     .agent(openai::GPT_5_2)
//!     .preamble("You are a dictionary assistant. Use the provided context documents.")
//!     .add_hook(DictionaryRag)
//!     .build();
//!
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
pub use hook::{
    AgentHook, Flow, HookContext, HookStack, InvalidToolCallContext, InvalidToolCallHookAction,
    RequestPatch, RunId, Scratchpad, StepEvent, StepEventKind,
};
pub use prompt_request::streaming::{
    FinalResponse, MultiTurnStreamItem, StreamingError, StreamingPromptRequest, StreamingResult,
    stream_to_stdout,
};
pub use prompt_request::{
    CompletionCall, PromptRequest, PromptResponse, TypedPromptRequest, TypedPromptResponse,
};
pub use run::{AgentRun, AgentRunStep, ModelTurn, ModelTurnOutcome, OutputMode, PendingToolCall};
pub use runner::AgentRunner;
