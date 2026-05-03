//! Conversation memory: Rig-managed persistent conversation history for agents.
//!
//! Memory differs from existing agent context features:
//! - [`crate::agent::AgentBuilder::context`]: static documents always included in prompts.
//! - [`crate::agent::AgentBuilder::dynamic_context`]: RAG documents fetched from a vector store.
//! - [`crate::agent::prompt_request::PromptRequest::with_history`]: caller-managed message history.
//! - **Memory** (this module): Rig-managed history loaded and saved automatically per
//!   conversation id.
//!
//! # Example
//!
//! ```no_run
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! use rig::client::{CompletionClient, ProviderClient};
//! use rig::completion::Prompt;
//! use rig::memory::InMemoryConversationMemory;
//! use rig::providers::openai;
//!
//! let memory = InMemoryConversationMemory::new();
//!
//! let openai = openai::Client::from_env()?;
//! let agent = openai.agent("gpt-4o").memory(memory).build();
//!
//! agent.prompt("My name is Alice.")
//!     .conversation("thread-1")
//!     .await?;
//!
//! let answer = agent.prompt("What's my name?")
//!     .conversation("thread-1")
//!     .await?;
//! # Ok(()) }
//! ```
//!
//! Truncation, summarization, and other history-shaping policies live in the
//! `rig-memory` companion crate. To shape history inside the in-tree backend,
//! pass a closure to [`InMemoryConversationMemory::with_filter`].

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use crate::{
    completion::Message,
    wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync},
};

/// Boxed error source for memory backend failures.
#[cfg(not(target_family = "wasm"))]
pub type MemoryBackendError = Box<dyn std::error::Error + Send + Sync + 'static>;

/// Boxed error source for memory backend failures.
#[cfg(target_family = "wasm")]
pub type MemoryBackendError = Box<dyn std::error::Error + 'static>;

/// Errors produced by a [`ConversationMemory`] backend.
#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    /// The backing store failed to load, append, or clear messages.
    #[error("Memory backend error: {0}")]
    Backend(MemoryBackendError),

    /// A history-shaping filter or policy rejected the loaded history.
    #[error("Memory policy error: {0}")]
    Policy(String),
}

impl MemoryError {
    /// Wrap an arbitrary error from a backend implementation.
    pub fn backend<E>(source: E) -> Self
    where
        E: Into<MemoryBackendError>,
    {
        Self::Backend(source.into())
    }
}

/// A persistent conversation history backend.
///
/// Implementors store an ordered list of [`Message`]s per `conversation_id`. Rig
/// invokes [`ConversationMemory::load`] before sending a prompt and
/// [`ConversationMemory::append`] after a successful turn.
///
/// Implementations should keep `append` cheap; it runs inline before the agent
/// returns its response.
pub trait ConversationMemory: WasmCompatSend + WasmCompatSync {
    /// Load the full conversation history for `conversation_id`.
    ///
    /// Returns an empty `Vec` if the conversation has no stored messages.
    fn load<'a>(
        &'a self,
        conversation_id: &'a str,
    ) -> WasmBoxedFuture<'a, Result<Vec<Message>, MemoryError>>;

    /// Append `messages` to the conversation identified by `conversation_id`.
    ///
    /// Called after a successful agent turn with the user prompt, the assistant
    /// response, and any tool-call/tool-result pairs that occurred during the turn.
    fn append<'a>(
        &'a self,
        conversation_id: &'a str,
        messages: Vec<Message>,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>>;

    /// Remove all stored messages for `conversation_id`.
    fn clear<'a>(
        &'a self,
        conversation_id: &'a str,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>>;
}

/// A history-shaping closure applied during [`InMemoryConversationMemory::load`].
///
/// Implemented automatically for any closure with the right signature; the
/// trait exists to combine `Fn` with the WASM-compatible `Send`/`Sync` markers
/// in a single trait object.
pub trait MessageFilter:
    Fn(Vec<Message>) -> Vec<Message> + WasmCompatSend + WasmCompatSync
{
}

impl<F> MessageFilter for F where
    F: Fn(Vec<Message>) -> Vec<Message> + WasmCompatSend + WasmCompatSync
{
}

/// A simple thread-safe in-memory [`ConversationMemory`] backed by a `HashMap`.
///
/// Messages are stored in process memory only and lost on restart. Useful for
/// tests, examples, and short-lived agents. Pass a closure to
/// [`InMemoryConversationMemory::with_filter`] to apply a history-shaping
/// transformation on every load (truncation, summarization, re-ordering, etc.).
/// Reusable named policies live in the `rig-memory` companion crate.
#[derive(Clone, Default)]
pub struct InMemoryConversationMemory {
    inner: Arc<Mutex<HashMap<String, Vec<Message>>>>,
    filter: Option<Arc<dyn MessageFilter>>,
}

impl InMemoryConversationMemory {
    /// Create an empty in-memory store with no filter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply `filter` to the loaded message list on every `load`.
    ///
    /// The filter runs after raw messages are read from the store and before
    /// they are returned to the agent. Use it for truncation, summarization, or
    /// any other shaping. For reusable named policies, depend on `rig-memory`.
    pub fn with_filter<F>(mut self, filter: F) -> Self
    where
        F: MessageFilter + 'static,
    {
        self.filter = Some(Arc::new(filter));
        self
    }

    fn lock(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, HashMap<String, Vec<Message>>>, MemoryError> {
        self.inner
            .lock()
            .map_err(|e| MemoryError::backend(std::io::Error::other(e.to_string())))
    }
}

impl std::fmt::Debug for InMemoryConversationMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InMemoryConversationMemory")
            .field("filter", &self.filter.as_ref().map(|_| "<filter>"))
            .finish()
    }
}

impl ConversationMemory for InMemoryConversationMemory {
    fn load<'a>(
        &'a self,
        conversation_id: &'a str,
    ) -> WasmBoxedFuture<'a, Result<Vec<Message>, MemoryError>> {
        Box::pin(async move {
            let messages = {
                let guard = self.lock()?;
                guard.get(conversation_id).cloned().unwrap_or_default()
            };
            match &self.filter {
                Some(filter) => Ok(filter(messages)),
                None => Ok(messages),
            }
        })
    }

    fn append<'a>(
        &'a self,
        conversation_id: &'a str,
        messages: Vec<Message>,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
        Box::pin(async move {
            let mut guard = self.lock()?;
            guard
                .entry(conversation_id.to_string())
                .or_default()
                .extend(messages);
            Ok(())
        })
    }

    fn clear<'a>(
        &'a self,
        conversation_id: &'a str,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
        Box::pin(async move {
            let mut guard = self.lock()?;
            guard.remove(conversation_id);
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::completion::Message;

    fn user(text: &str) -> Message {
        Message::user(text)
    }

    fn assistant(text: &str) -> Message {
        Message::assistant(text)
    }

    #[tokio::test]
    async fn round_trip() {
        let mem = InMemoryConversationMemory::new();
        assert!(mem.load("c1").await.unwrap().is_empty());

        mem.append("c1", vec![user("hello"), assistant("hi")])
            .await
            .unwrap();

        let loaded = mem.load("c1").await.unwrap();
        assert_eq!(loaded.len(), 2);
    }

    #[tokio::test]
    async fn isolation_between_conversations() {
        let mem = InMemoryConversationMemory::new();
        mem.append("a", vec![user("hi a")]).await.unwrap();
        mem.append("b", vec![user("hi b")]).await.unwrap();

        assert_eq!(mem.load("a").await.unwrap().len(), 1);
        assert_eq!(mem.load("b").await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn clear_removes_history() {
        let mem = InMemoryConversationMemory::new();
        mem.append("c", vec![user("x")]).await.unwrap();
        mem.clear("c").await.unwrap();
        assert!(mem.load("c").await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn with_filter_transforms_loaded_messages() {
        let mem = InMemoryConversationMemory::new()
            .with_filter(|msgs: Vec<Message>| msgs.into_iter().rev().take(2).collect());

        mem.append(
            "c",
            vec![user("1"), assistant("2"), user("3"), assistant("4")],
        )
        .await
        .unwrap();

        let loaded = mem.load("c").await.unwrap();
        assert_eq!(loaded.len(), 2, "filter should retain only 2 messages");
    }
}
