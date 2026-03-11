//! Pluggable conversation memory for [`Agent`].
//!
//! The [`ConversationMemory`] trait lets you swap the default (stateless) chat
//! history with any storage backend — an in-process ring buffer, a Redis list,
//! a vector store, or a summarising compressor.
//!
//! # Built-in implementations
//!
//! | Type | Behaviour |
//! |---|---|
//! | [`InMemoryConversation`] | Keeps the last *N* messages in a `VecDeque`. |
//! | [`NoMemory`] | Discards all messages (stateless default). |
//!
//! # Example
//!
//! ```rust,ignore
//! use rig::memory::{ConversationMemory, InMemoryConversation};
//! use rig::completion::Message;
//!
//! #[tokio::main]
//! async fn main() {
//!     let memory = InMemoryConversation::new(20); // keep last 20 messages
//!
//!     memory.push(Message::user("What is backpressure?")).await;
//!     memory.push(Message::assistant("Backpressure is …")).await;
//!
//!     let history = memory.history().await;
//!     assert_eq!(history.len(), 2);
//! }
//! ```
//!
//! # Implementing a custom backend
//!
//! ```rust,ignore
//! use async_trait::async_trait;
//! use rig::memory::ConversationMemory;
//! use rig::completion::Message;
//!
//! struct RedisMemory { /* ... */ }
//!
//! #[async_trait]
//! impl ConversationMemory for RedisMemory {
//!     async fn push(&self, message: Message) { /* LPUSH to Redis */ }
//!     async fn history(&self) -> Vec<Message> { /* LRANGE from Redis */ vec![] }
//!     async fn clear(&self) { /* DEL key */ }
//!     async fn len(&self) -> usize { /* LLEN */ 0 }
//! }
//! ```
//!
//! # Attribution
//!
//! Designed and implemented by Matthew Busel.

use std::collections::VecDeque;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;

use crate::completion::Message;

// ── Trait ────────────────────────────────────────────────────────────────────

/// Pluggable storage for conversation history.
///
/// All mutating methods take `&self` rather than `&mut self` so that
/// implementations can be wrapped in [`Arc`] and shared across agent handles.
#[async_trait]
pub trait ConversationMemory: Send + Sync {
    /// Append `message` to the end of the conversation.
    async fn push(&self, message: Message);

    /// Return all stored messages in chronological order.
    async fn history(&self) -> Vec<Message>;

    /// Discard all stored messages.
    async fn clear(&self);

    /// Number of messages currently stored.
    async fn len(&self) -> usize;

    /// Returns `true` if no messages are stored.
    async fn is_empty(&self) -> bool {
        self.len().await == 0
    }
}

// ── InMemoryConversation ─────────────────────────────────────────────────────

/// Stores the most recent `capacity` messages in a bounded `VecDeque`.
///
/// When `capacity` is reached, the oldest message is evicted before the new
/// one is appended, keeping memory use bounded regardless of conversation
/// length.
///
/// Pass `capacity = 0` for an unbounded store, or use
/// [`InMemoryConversation::unbounded`].
///
/// Cloning creates an independent copy (O(n)). Wrap in [`Arc`] for
/// shared ownership across tasks.
#[derive(Debug)]
pub struct InMemoryConversation {
    inner: Arc<RwLock<VecDeque<Message>>>,
    capacity: usize,
}

impl Default for InMemoryConversation {
    fn default() -> Self {
        Self::unbounded()
    }
}

impl InMemoryConversation {
    /// Create a store that keeps at most `capacity` messages.
    ///
    /// `capacity = 0` means unbounded.
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(VecDeque::new())),
            capacity,
        }
    }

    /// Create an unbounded store that retains every message.
    pub fn unbounded() -> Self {
        Self::new(0)
    }
}

#[async_trait]
impl ConversationMemory for InMemoryConversation {
    async fn push(&self, message: Message) {
        let mut guard = self.inner.write().await;
        if self.capacity > 0 && guard.len() >= self.capacity {
            guard.pop_front();
        }
        guard.push_back(message);
    }

    async fn history(&self) -> Vec<Message> {
        self.inner.read().await.iter().cloned().collect()
    }

    async fn clear(&self) {
        self.inner.write().await.clear();
    }

    async fn len(&self) -> usize {
        self.inner.read().await.len()
    }
}

/// A sliding-window memory store — alias for [`InMemoryConversation`].
pub type SlidingWindowMemory = InMemoryConversation;

// ── NoMemory ─────────────────────────────────────────────────────────────────

/// A no-op memory implementation that discards every message.
///
/// This matches the current `Agent` behaviour (stateless per call) and is
/// useful as a default type parameter when memory is optional.
#[derive(Clone, Debug, Default)]
pub struct NoMemory;

#[async_trait]
impl ConversationMemory for NoMemory {
    async fn push(&self, _: Message) {}
    async fn history(&self) -> Vec<Message> { vec![] }
    async fn clear(&self) {}
    async fn len(&self) -> usize { 0 }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn user_msg(text: &str) -> Message {
        Message::user(text)
    }

    #[tokio::test]
    async fn push_and_retrieve() {
        let mem = InMemoryConversation::new(10);
        mem.push(user_msg("hello")).await;
        mem.push(user_msg("world")).await;
        assert_eq!(mem.history().await.len(), 2);
    }

    #[tokio::test]
    async fn capacity_evicts_oldest() {
        let mem = InMemoryConversation::new(3);
        for i in 0..5u8 {
            mem.push(user_msg(&i.to_string())).await;
        }
        assert_eq!(mem.len().await, 3, "must not exceed capacity");
    }

    #[tokio::test]
    async fn clear_empties_store() {
        let mem = InMemoryConversation::new(10);
        mem.push(user_msg("a")).await;
        mem.clear().await;
        assert!(mem.is_empty().await);
    }

    #[tokio::test]
    async fn unbounded_keeps_all() {
        let mem = InMemoryConversation::unbounded();
        for i in 0..200u16 {
            mem.push(user_msg(&i.to_string())).await;
        }
        assert_eq!(mem.len().await, 200);
    }

    #[tokio::test]
    async fn no_memory_always_empty() {
        let mem = NoMemory;
        mem.push(user_msg("ignored")).await;
        assert!(mem.history().await.is_empty());
    }

    #[tokio::test]
    async fn shared_via_arc() {
        let mem = Arc::new(InMemoryConversation::new(5));
        let mem2 = Arc::clone(&mem);
        mem.push(user_msg("from task 1")).await;
        mem2.push(user_msg("from task 2")).await;
        assert_eq!(mem.len().await, 2);
    }
}
