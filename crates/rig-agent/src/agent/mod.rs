//! This module contains the implementation of the [Agent] struct and its builder.
//!
//! The [Agent] struct represents an LLM agent, which combines an LLM model with a preamble (system prompt),
//! a set of static context documents, and a set of tools. Every registered
//! tool is advertised; hooks can narrow the advertised set per turn with
//! [`RequestPatch::active_tools`].
//!
//! The [Agent] struct is highly configurable, allowing the user to define anything from
//! a simple bot with a specific system prompt to a complex RAG system.
//!
//! All agent execution goes through the session drivers
//! ([`AgentSession`](crate::session::AgentSession) and
//! [`AgentStream`](crate::stream::AgentStream)), so hooks and lifecycle
//! policies cannot be bypassed through a raw agent request builder.
//!
//! The [AgentBuilder] implements the builder pattern for creating instances of [Agent].
//! It allows configuring the model, preamble, context documents, tools, temperature, and additional parameters
//! before building the agent.
//!
//! # Example
//! ```no_run
//! use rig_agent::prelude::*;
//! use rig_core::providers::openai;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! // Configure the agent. A provider is plain data: a `functions::Config`
//! // wrapped in the `ProviderConfig` arm that names it.
//! let agent = AgentBuilder::new(ProviderConfig::OpenAi(
//!         openai::functions::Config::from_env(openai::GPT_5_2)?,
//!     ))
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
//! // Per-run overrides live on the per-request runner.
//! let response = agent.runner("Prompt").temperature(0.9).run().await?;
//! # Ok(())
//! # }
//! ```
//!
//! Passive RAG is a hook recipe: attach a [`HookEntry`](crate::hooks::HookEntry),
//! embed the prompt on
//! every completion call, query a concrete vector store, and inject the hits
//! with [`RequestPatch::extra_context`]. Because it is an ordinary hook, custom
//! query selection, filtering, reranking, caching, formatting, and failure
//! handling are all in the application's hands, and it composes with other
//! hooks in registration order. Active RAG instead exposes the store as a
//! custom tool so the model decides when to search (see the
//! `complex_agentic_loop_claude` example).
//!
//! Passive RAG agent example
//! ```no_run
//! use rig_agent::{
//!     agent::{CompletionCallAction, RequestPatch},
//!     completion::Document,
//!     hooks::{HookDecision, HookEntry, HookEvent},
//!     prelude::*,
//!     provider::{EmbedderConfig, Runtime, embed},
//! };
//! use rig_core::{
//!     providers::openai,
//!     vector_store::{StoreRecord, VectorSearchRequest, in_memory_store::InMemoryVectorStore},
//! };
//! use std::sync::Arc;
//!
//! /// Embeds each prompt, searches the store, and appends the hits as
//! /// per-turn context documents. A retrieval failure stops the run before
//! /// provider I/O.
//! fn rag_hook(
//!     embedder: EmbedderConfig,
//!     rt: Arc<Runtime>,
//!     store: InMemoryVectorStore,
//!     samples: u64,
//! ) -> HookEntry {
//!     HookEntry::with_state("rag", (embedder, rt, store, samples), |state, event| async move {
//!             let HookEvent::BeforeModelCall { prompt, history, .. } = event else {
//!                 return HookDecision::Continue;
//!             };
//!             let (embedder, rt, store, samples) = state.as_ref();
//!             // Search with the prompt's text, falling back to the latest
//!             // textual history message.
//!             let query = prompt.rag_text().or_else(|| {
//!                 history.iter().rev().find_map(|message| message.rag_text())
//!             });
//!             let Some(query) = query else {
//!                 return HookDecision::CompletionCall(CompletionCallAction::continue_run());
//!             };
//!
//!             // Embed the query, then run a pre-embedded search.
//!             let embedded = match embed(embedder, rt, vec![query]).await {
//!                 Ok(response) => response.embeddings.into_iter().next(),
//!                 Err(error) => {
//!                     return HookDecision::CompletionCall(
//!                         CompletionCallAction::stop(error.to_string()),
//!                     );
//!                 }
//!             };
//!             let Some(embedded) = embedded else {
//!                 return HookDecision::CompletionCall(CompletionCallAction::stop(
//!                     "empty embedding response",
//!                 ));
//!             };
//!             let request = VectorSearchRequest::new(rig_core::OneOrMany::one(embedded), *samples);
//!             match store.top_n(request).await {
//!                 Ok(hits) => HookDecision::CompletionCall(CompletionCallAction::patch(
//!                     RequestPatch::new().extra_context(hits.into_iter().map(|hit| Document {
//!                         id: hit.id,
//!                         text: hit.payload.to_string(),
//!                         additional_props: Default::default(),
//!                     })),
//!                 )),
//!                 Err(error) => HookDecision::CompletionCall(CompletionCallAction::stop(
//!                     error.to_string(),
//!                 )),
//!             }
//!     })
//! }
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let rt = Arc::new(Runtime::new());
//! let embedder = EmbedderConfig::OpenAi(openai::functions::EmbeddingConfig::new(
//!     openai::TEXT_EMBEDDING_3_SMALL,
//! ));
//!
//! // Embed the corpus up front and load it into a concrete store.
//! let corpus = vec![
//!     "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets".to_string(),
//!     "Definition of a *glarb-glarb*: A glarb-glarb is an ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.".to_string(),
//!     "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.".to_string(),
//! ];
//! let embedded = embed(&embedder, &rt, corpus.clone()).await?;
//! let store = InMemoryVectorStore::new();
//! store
//!     .insert(
//!         corpus
//!             .iter()
//!             .zip(embedded.embeddings)
//!             .enumerate()
//!             .map(|(i, (text, embedding))| StoreRecord {
//!                 id: format!("doc{i}"),
//!                 payload: serde_json::Value::String(text.clone()),
//!                 embeddings: rig_core::OneOrMany::one(embedding),
//!             })
//!             .collect(),
//!     )
//!     .await?;
//!
//! let agent = AgentBuilder::new(ProviderConfig::OpenAi(
//!         openai::functions::Config::from_env(openai::GPT_5_2)?,
//!     ))
//!     .preamble("
//!         You are a dictionary assistant here to assist the user in understanding the meaning of words.
//!         You will find additional non-standard word definitions that could be useful below.
//!     ")
//!     .add_hook(rag_hook(embedder, rt, store, 1))
//!     .build();
//!
//! // Prompt the agent and print the response
//! let response = agent.prompt("What does \"glarb-glarb\" mean?").await?;
//! # Ok(())
//! # }
//! ```
pub(crate) mod attempt;
mod builder;
mod completion;
pub mod config;
pub mod hook;
#[cfg(test)]
pub(crate) mod mock_support;
pub mod prepare;
pub mod request;
pub mod response;
pub mod run;
pub(crate) mod telemetry;

/// Fallback display name used in telemetry spans and logs when an agent has no
/// configured name.
pub(crate) const UNKNOWN_AGENT_NAME: &str = "Unnamed Agent";

pub use builder::AgentBuilder;
pub use completion::Agent;
pub use config::AgentConfig;
pub use hook::{
    CompletionCallAction, InvalidToolCallAction, InvalidToolCallContext, ModelTurnAction,
    ObservationAction, RequestPatch, ResolvedToolCall, ResolvedToolCallDisposition, RetryRequest,
    RunId, ToolCallAction, ToolCallResolution, ToolResultAction, ToolResultResolution,
    fold_completion_actions, fold_invalid_resolutions, fold_observation_actions,
};
pub use prepare::{PreparedModelAttempt, PreparedRequest, ToolCatalog, prepare_request};
pub use request::SessionRunner;
pub use response::{CompletionCall, PromptResponse};
pub use rig_core::message::Text;
pub use run::{
    AcceptedModelTurn, AgentRun, AgentRunStep, ModelAttemptId, ModelTurn, ModelTurnOutcome,
    OutputMode, PendingToolCall, ToolInvocationDisposition,
};
