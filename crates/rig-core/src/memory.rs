//! Conversation memory: a concrete in-process conversation store plus the
//! error type shared by memory-shaped host code.
//!
//! Memory is **host-owned data**, not an agent behavior slot. Nothing in
//! `rig-core` or `rig-agent` loads or saves conversation history on your
//! behalf: the host calls [`InMemoryConversationMemory::load`] before a run
//! and [`InMemoryConversationMemory::append`] after it (see
//! `rig_agent::agent_api` module docs for the canonical recipe, including the
//! exact failure semantics the classic driver used).
//!
//! Memory differs from the other ways history reaches a model:
//! - static context: documents always included in prompts;
//! - per-turn request patches supplied by hooks;
//! - caller-managed message history supplied directly on completion requests;
//! - **memory** (this module): history the host persists per conversation id.
//!
//! # Example
//!
//! ```
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! use rig_core::{completion::Message, memory::InMemoryConversationMemory};
//!
//! let memory = InMemoryConversationMemory::new();
//! memory.append(
//!     "thread-1",
//!     vec![
//!         Message::user("My name is Alice."),
//!         Message::assistant("Hello, Alice!"),
//!     ],
//! )?;
//! let history = memory.load("thread-1")?;
//! assert_eq!(history.len(), 2);
//! # Ok(()) }
//! # run().unwrap();
//! ```
//!
//! Truncation, demotion, and summarization policies live in the `rig-memory`
//! companion crate as plain data (`MemoryPolicy`, `Compactor`) plus a
//! concrete `PolicyMemory` whose `append` hands the host owned outcome
//! events. Shape history by running a policy yourself before you append or
//! after you load — there is no filter callback stored in the store.
//!
//! # Where memory code lives
//!
//! `rig-memory` is the canonical companion crate for everything
//! memory-related: it re-exports every item in this module and adds the
//! reusable policies. The store remains *defined* here because `rig-memory`
//! and `rig-agent` both depend on `rig-core`.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use crate::completion::Message;

/// Boxed error source for memory backend failures.
#[cfg(not(target_family = "wasm"))]
pub type MemoryBackendError = Box<dyn std::error::Error + Send + Sync + 'static>;

/// Boxed error source for memory backend failures.
#[cfg(target_family = "wasm")]
pub type MemoryBackendError = Box<dyn std::error::Error + 'static>;

/// Errors produced by conversation-memory code.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum MemoryError {
    /// A conversation store failed to load, append, or clear messages. Host
    /// stores (databases, files, remote services) report their own failures
    /// through this variant.
    #[error("Memory backend error: {0}")]
    Backend(MemoryBackendError),

    /// An internal invariant was violated (e.g. a poisoned in-process lock).
    /// Distinct from [`MemoryError::Backend`], which is reserved for failures
    /// of the underlying conversation store.
    #[error("Memory internal error: {0}")]
    Internal(String),
}

impl MemoryError {
    /// Wrap an arbitrary error from a store implementation.
    pub fn backend<E>(source: E) -> Self
    where
        E: Into<MemoryBackendError>,
    {
        Self::Backend(source.into())
    }
}

/// A simple thread-safe in-process conversation store backed by a `HashMap`.
///
/// Messages are stored in process memory only and lost on restart. Useful for
/// tests, examples, and short-lived agents. Cloning shares the same storage.
/// All methods are plain inherent methods: the store is a `Mutex<HashMap>`,
/// so nothing here needs to be `async`.
///
/// To shape history, run a `rig_memory::MemoryPolicy` over the loaded
/// messages yourself, or use `rig_memory::PolicyMemory`, which wraps this
/// store with a policy and reports demotion/compaction as owned data.
#[derive(Clone, Default, Debug)]
pub struct InMemoryConversationMemory {
    inner: Arc<Mutex<HashMap<String, Vec<Message>>>>,
}

impl InMemoryConversationMemory {
    /// Create an empty in-memory store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Load the full stored history for `conversation_id`.
    ///
    /// Returns an empty `Vec` if the conversation has no stored messages.
    pub fn load(&self, conversation_id: &str) -> Result<Vec<Message>, MemoryError> {
        let guard = self.lock()?;
        Ok(guard.get(conversation_id).cloned().unwrap_or_default())
    }

    /// Append `messages` to the conversation identified by `conversation_id`.
    pub fn append(&self, conversation_id: &str, messages: Vec<Message>) -> Result<(), MemoryError> {
        let mut guard = self.lock()?;
        guard
            .entry(conversation_id.to_string())
            .or_default()
            .extend(messages);
        Ok(())
    }

    /// Remove all stored messages for `conversation_id`.
    pub fn clear(&self, conversation_id: &str) -> Result<(), MemoryError> {
        let mut guard = self.lock()?;
        guard.remove(conversation_id);
        Ok(())
    }

    /// Number of conversations currently stored. Useful for telemetry and
    /// leak detection in tests.
    pub fn tracked_conversations(&self) -> Result<usize, MemoryError> {
        Ok(self.lock()?.len())
    }

    fn lock(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, HashMap<String, Vec<Message>>>, MemoryError> {
        self.inner
            .lock()
            .map_err(|e| MemoryError::Internal(e.to_string()))
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

    #[test]
    fn round_trip() {
        let mem = InMemoryConversationMemory::new();
        assert!(mem.load("c1").unwrap().is_empty());

        mem.append("c1", vec![user("hello"), assistant("hi")])
            .unwrap();

        let loaded = mem.load("c1").unwrap();
        assert_eq!(loaded.len(), 2);
    }

    #[test]
    fn isolation_between_conversations() {
        let mem = InMemoryConversationMemory::new();
        mem.append("a", vec![user("hi a")]).unwrap();
        mem.append("b", vec![user("hi b")]).unwrap();

        assert_eq!(mem.load("a").unwrap().len(), 1);
        assert_eq!(mem.load("b").unwrap().len(), 1);
        assert_eq!(mem.tracked_conversations().unwrap(), 2);
    }

    #[test]
    fn clear_removes_history() {
        let mem = InMemoryConversationMemory::new();
        mem.append("c", vec![user("x")]).unwrap();
        mem.clear("c").unwrap();
        assert!(mem.load("c").unwrap().is_empty());
    }

    #[test]
    fn clones_share_storage() {
        let mem = InMemoryConversationMemory::new();
        let handle = mem.clone();

        handle.append("c", vec![user("hello")]).unwrap();

        assert_eq!(mem.load("c").unwrap().len(), 1);
        mem.clear("c").unwrap();
        assert!(handle.load("c").unwrap().is_empty());
    }
}
