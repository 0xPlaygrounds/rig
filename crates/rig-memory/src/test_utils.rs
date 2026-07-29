//! Conversation-store test doubles for deterministic host-side memory tests.
//!
//! Memory is a host concern: nothing in the agent layer calls these. They
//! exist so host code that follows the load-before / append-after recipe can
//! be tested deterministically (call counts, load failures, append failures).
//! Enable the `test-utils` feature of this crate to use them.

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use rig_core::{
    completion::Message,
    memory::{InMemoryConversationMemory, MemoryError},
};

/// Store that records load and append calls while delegating storage to
/// [`InMemoryConversationMemory`].
#[derive(Clone, Default, Debug)]
pub struct CountingMemory {
    inner: InMemoryConversationMemory,
    loads: Arc<AtomicUsize>,
    appends: Arc<AtomicUsize>,
}

impl CountingMemory {
    /// Return the backing in-memory store. Loads and appends performed
    /// directly on it are not counted.
    pub fn inner(&self) -> &InMemoryConversationMemory {
        &self.inner
    }

    /// Return the number of calls to [`CountingMemory::load`].
    pub fn load_count(&self) -> usize {
        self.loads.load(Ordering::SeqCst)
    }

    /// Return the number of calls to [`CountingMemory::append`].
    pub fn append_count(&self) -> usize {
        self.appends.load(Ordering::SeqCst)
    }

    /// Load the stored history, counting the call.
    pub fn load(&self, conversation_id: &str) -> Result<Vec<Message>, MemoryError> {
        self.loads.fetch_add(1, Ordering::SeqCst);
        self.inner.load(conversation_id)
    }

    /// Append messages, counting the call.
    pub fn append(&self, conversation_id: &str, messages: Vec<Message>) -> Result<(), MemoryError> {
        self.appends.fetch_add(1, Ordering::SeqCst);
        self.inner.append(conversation_id, messages)
    }

    /// Clear the stored history. Not counted.
    pub fn clear(&self, conversation_id: &str) -> Result<(), MemoryError> {
        self.inner.clear(conversation_id)
    }
}

/// Store that always fails on load and no-ops append and clear.
#[derive(Clone, Debug)]
pub struct FailingMemory {
    message: String,
}

impl FailingMemory {
    /// Create a load-failing store.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// Always fails.
    pub fn load(&self, _conversation_id: &str) -> Result<Vec<Message>, MemoryError> {
        Err(MemoryError::backend(std::io::Error::other(
            self.message.clone(),
        )))
    }

    /// Always succeeds.
    pub fn append(
        &self,
        _conversation_id: &str,
        _messages: Vec<Message>,
    ) -> Result<(), MemoryError> {
        Ok(())
    }

    /// Always succeeds.
    pub fn clear(&self, _conversation_id: &str) -> Result<(), MemoryError> {
        Ok(())
    }
}

impl Default for FailingMemory {
    fn default() -> Self {
        Self::new("load boom")
    }
}

/// Store that loads empty history and always fails on append.
#[derive(Clone, Debug)]
pub struct AppendFailingMemory {
    message: String,
}

impl AppendFailingMemory {
    /// Create an append-failing store.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// Always returns an empty history.
    pub fn load(&self, _conversation_id: &str) -> Result<Vec<Message>, MemoryError> {
        Ok(Vec::new())
    }

    /// Always fails.
    pub fn append(
        &self,
        _conversation_id: &str,
        _messages: Vec<Message>,
    ) -> Result<(), MemoryError> {
        Err(MemoryError::backend(std::io::Error::other(
            self.message.clone(),
        )))
    }

    /// Always succeeds.
    pub fn clear(&self, _conversation_id: &str) -> Result<(), MemoryError> {
        Ok(())
    }
}

impl Default for AppendFailingMemory {
    fn default() -> Self {
        Self::new("append boom")
    }
}
