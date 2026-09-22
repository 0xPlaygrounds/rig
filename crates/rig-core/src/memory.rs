//! Conversation history storage, filtering, and compaction interfaces.
//!
//! ```no_run
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! use rig_core::{
//!     completion::Message,
//!     memory::{ConversationMemory, InMemoryConversationMemory},
//! };
//!
//! let memory = InMemoryConversationMemory::new();
//! memory
//!     .append(
//!         &"thread-1".into(),
//!         vec![
//!             Message::user("My name is Alice."),
//!             Message::assistant("Hello, Alice!"),
//!         ],
//!     )
//!     .await?;
//! let history = memory.load(&"thread-1".into()).await?;
//! assert_eq!(history.len(), 2);
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

use crate::id::ConversationId;
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
    Backend(#[source] MemoryBackendError),

    /// A history-shaping filter or policy rejected the loaded history.
    #[error("Memory policy error: {0}")]
    Policy(String),

    /// An internal invariant was violated (e.g. a poisoned in-process lock).
    /// Distinct from [`MemoryError::Backend`], which is reserved for failures
    /// of the underlying conversation store.
    #[error("Memory internal error: {0}")]
    Internal(String),
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
/// runtimes invoke [`ConversationMemory::load`] before sending a prompt and
/// [`ConversationMemory::append`] after a successful run.
///
/// Appends run inline before the agent returns its response. Load failures
/// prevent model calls; append failures are reported alongside the successful
/// answer without retry. Writes are not transactional or exactly-once: an
/// error may occur after the backend has persisted messages.
pub trait ConversationMemory: WasmCompatSend + WasmCompatSync {
    /// Load the full conversation history for `conversation_id`.
    ///
    /// Returns an empty `Vec` if the conversation has no stored messages.
    fn load<'a>(
        &'a self,
        conversation_id: &'a ConversationId,
    ) -> WasmBoxedFuture<'a, Result<Vec<Message>, MemoryError>>;

    /// Append `messages` to the conversation identified by `conversation_id`.
    ///
    /// Called after a successful agent turn with the user prompt, the assistant
    /// response, and any tool-call/tool-result pairs that occurred during the turn.
    fn append<'a>(
        &'a self,
        conversation_id: &'a ConversationId,
        messages: Vec<Message>,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>>;

    /// Remove all stored messages for `conversation_id`.
    fn clear<'a>(
        &'a self,
        conversation_id: &'a ConversationId,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>>;
}

macro_rules! forward_memory_trait {
    (ConversationMemory: $($ptr:ident)+) => {$(
        impl<M> ConversationMemory for $ptr<M>
        where
            M: ConversationMemory + ?Sized,
        {
            fn load<'a>(
                &'a self,
                conversation_id: &'a ConversationId,
            ) -> WasmBoxedFuture<'a, Result<Vec<Message>, MemoryError>> {
                (**self).load(conversation_id)
            }

            fn append<'a>(
                &'a self,
                conversation_id: &'a ConversationId,
                messages: Vec<Message>,
            ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
                (**self).append(conversation_id, messages)
            }

            fn clear<'a>(
                &'a self,
                conversation_id: &'a ConversationId,
            ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
                (**self).clear(conversation_id)
            }
        }
    )+};
    (DemotionHook: $($ptr:ident)+) => {$(
        impl<H> DemotionHook for $ptr<H>
        where
            H: DemotionHook + ?Sized,
        {
            fn on_demote<'a>(
                &'a self,
                conversation_id: &'a ConversationId,
                messages: Vec<Message>,
            ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
                (**self).on_demote(conversation_id, messages)
            }
        }
    )+};
    (Compactor: $($ptr:ident)+) => {$(
        impl<C> Compactor for $ptr<C>
        where
            C: Compactor + ?Sized,
        {
            type Artifact = C::Artifact;

            fn compact<'a>(
                &'a self,
                conversation_id: &'a ConversationId,
                evicted: &'a [Message],
                carry_over: Option<&'a Self::Artifact>,
            ) -> WasmBoxedFuture<'a, Result<Self::Artifact, MemoryError>> {
                (**self).compact(conversation_id, evicted, carry_over)
            }
        }
    )+};
}

forward_memory_trait!(ConversationMemory: Arc Box);

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

/// Receives messages removed from active history during [`ConversationMemory::load`].
/// Hooks are awaited inline, so their latency delays the next turn.
///
/// Implementations must be idempotent on `(conversation_id, messages)`.
/// Adapter delivery watermarks are not persisted; restarts or newly constructed
/// adapters may redeliver messages. Durable hooks should deduplicate with a
/// stable key such as a conversation ID and content hash.
pub trait DemotionHook: WasmCompatSend + WasmCompatSync {
    /// Receive `messages` that were demoted out of the active window for
    /// `conversation_id`.
    ///
    /// `messages` are in original conversation order. Errors are propagated
    /// as [`MemoryError::Backend`] by the composing adapter.
    fn on_demote<'a>(
        &'a self,
        conversation_id: &'a ConversationId,
        messages: Vec<Message>,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>>;
}

/// A [`DemotionHook`] that does nothing. Useful as a default when an adapter
/// requires a hook value but the caller has no long-tail store wired up yet.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopDemotionHook;

impl DemotionHook for NoopDemotionHook {
    fn on_demote<'a>(
        &'a self,
        _conversation_id: &'a ConversationId,
        _messages: Vec<Message>,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
        Box::pin(async move { Ok(()) })
    }
}

forward_memory_trait!(DemotionHook: Arc);

/// Derives an artifact from evicted messages for insertion before recent history.
/// Compaction runs inline during loading and delays the next turn.
///
/// `carry_over` contains the previous artifact, if any. Combine it with the
/// evicted messages to preserve earlier context in a rolling summary, or ignore
/// it to summarize only the newly evicted messages.
///
/// Delivery watermarks are not persisted across restarts. Implementations with
/// side effects should deduplicate by conversation ID and content hash.
pub trait Compactor: WasmCompatSend + WasmCompatSync {
    /// Summary convertible to a history message and clonable for the next
    /// compaction's `carry_over`.
    type Artifact: Into<Message> + Clone + WasmCompatSend + WasmCompatSync + 'static;

    /// Produce a summary artifact for `evicted`, optionally combining it
    /// with the previous summary in `carry_over`.
    ///
    /// `evicted` is in original conversation order. Errors are propagated
    /// unchanged by composing adapters; pick the [`MemoryError`] variant
    /// that best describes the failure ([`MemoryError::Backend`] for I/O
    /// or remote-LLM faults, [`MemoryError::Internal`] for invariant
    /// breaks, and so on). The adapter does not re-wrap the returned
    /// variant.
    fn compact<'a>(
        &'a self,
        conversation_id: &'a ConversationId,
        evicted: &'a [Message],
        carry_over: Option<&'a Self::Artifact>,
    ) -> WasmBoxedFuture<'a, Result<Self::Artifact, MemoryError>>;
}

forward_memory_trait!(Compactor: Arc);

/// A simple thread-safe in-memory [`ConversationMemory`] backed by a `HashMap`.
///
/// Messages are stored in process memory only and lost on restart. Useful for
/// tests, examples, and short-lived agents. Pass a closure to
/// [`InMemoryConversationMemory::with_filter`] to apply a history-shaping
/// transformation on every load (truncation, summarization, re-ordering, etc.).
/// Reusable named policies live in the `rig-memory` companion crate.
#[derive(Clone, Default)]
pub struct InMemoryConversationMemory {
    inner: Arc<Mutex<HashMap<ConversationId, Vec<Message>>>>,
    filter: Option<Arc<dyn MessageFilter>>,
}

impl InMemoryConversationMemory {
    /// Create an empty in-memory store with no filter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces the filter applied to each loaded history after releasing the
    /// store lock. Filtering does not modify stored messages.
    pub fn with_filter<F>(mut self, filter: F) -> Self
    where
        F: MessageFilter + 'static,
    {
        self.filter = Some(Arc::new(filter));
        self
    }

    fn lock(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, HashMap<ConversationId, Vec<Message>>>, MemoryError> {
        self.inner
            .lock()
            .map_err(|e| MemoryError::Internal(e.to_string()))
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
        conversation_id: &'a ConversationId,
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
        conversation_id: &'a ConversationId,
        messages: Vec<Message>,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
        Box::pin(async move {
            let mut guard = self.lock()?;
            guard
                .entry(conversation_id.clone())
                .or_default()
                .extend(messages);
            Ok(())
        })
    }

    fn clear<'a>(
        &'a self,
        conversation_id: &'a ConversationId,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
        Box::pin(async move {
            let mut guard = self.lock()?;
            guard.remove(conversation_id);
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests;
