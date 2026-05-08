//! Conversation memory helpers for deterministic agent tests.

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use crate::{
    completion::Message,
    memory::{ConversationMemory, InMemoryConversationMemory, MemoryError},
    wasm_compat::WasmBoxedFuture,
};

/// Memory backend that records load and append calls while delegating storage to
/// [`InMemoryConversationMemory`].
#[derive(Clone, Default)]
pub struct CountingMemory {
    inner: InMemoryConversationMemory,
    loads: Arc<AtomicUsize>,
    appends: Arc<AtomicUsize>,
}

impl CountingMemory {
    /// Return the backing in-memory store.
    pub fn inner(&self) -> &InMemoryConversationMemory {
        &self.inner
    }

    /// Return the number of calls to [`ConversationMemory::load`].
    pub fn load_count(&self) -> usize {
        self.loads.load(Ordering::SeqCst)
    }

    /// Return the number of calls to [`ConversationMemory::append`].
    pub fn append_count(&self) -> usize {
        self.appends.load(Ordering::SeqCst)
    }
}

impl ConversationMemory for CountingMemory {
    fn load<'a>(
        &'a self,
        conversation_id: &'a str,
    ) -> WasmBoxedFuture<'a, Result<Vec<Message>, MemoryError>> {
        self.loads.fetch_add(1, Ordering::SeqCst);
        self.inner.load(conversation_id)
    }

    fn append<'a>(
        &'a self,
        conversation_id: &'a str,
        messages: Vec<Message>,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
        self.appends.fetch_add(1, Ordering::SeqCst);
        self.inner.append(conversation_id, messages)
    }

    fn clear<'a>(
        &'a self,
        conversation_id: &'a str,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
        self.inner.clear(conversation_id)
    }
}

/// Memory backend that always fails on load and no-ops append and clear.
#[derive(Clone)]
pub struct FailingMemory {
    message: String,
}

impl FailingMemory {
    /// Create a load-failing memory backend.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Default for FailingMemory {
    fn default() -> Self {
        Self::new("load boom")
    }
}

impl ConversationMemory for FailingMemory {
    fn load<'a>(
        &'a self,
        _conversation_id: &'a str,
    ) -> WasmBoxedFuture<'a, Result<Vec<Message>, MemoryError>> {
        let message = self.message.clone();
        Box::pin(async move { Err(MemoryError::backend(std::io::Error::other(message))) })
    }

    fn append<'a>(
        &'a self,
        _conversation_id: &'a str,
        _messages: Vec<Message>,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
        Box::pin(async { Ok(()) })
    }

    fn clear<'a>(
        &'a self,
        _conversation_id: &'a str,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
        Box::pin(async { Ok(()) })
    }
}

/// Memory backend that loads empty history and always fails on append.
#[derive(Clone)]
pub struct AppendFailingMemory {
    message: String,
}

impl AppendFailingMemory {
    /// Create an append-failing memory backend.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Default for AppendFailingMemory {
    fn default() -> Self {
        Self::new("append boom")
    }
}

impl ConversationMemory for AppendFailingMemory {
    fn load<'a>(
        &'a self,
        _conversation_id: &'a str,
    ) -> WasmBoxedFuture<'a, Result<Vec<Message>, MemoryError>> {
        Box::pin(async { Ok(Vec::new()) })
    }

    fn append<'a>(
        &'a self,
        _conversation_id: &'a str,
        _messages: Vec<Message>,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
        let message = self.message.clone();
        Box::pin(async move { Err(MemoryError::backend(std::io::Error::other(message))) })
    }

    fn clear<'a>(
        &'a self,
        _conversation_id: &'a str,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
        Box::pin(async { Ok(()) })
    }
}
