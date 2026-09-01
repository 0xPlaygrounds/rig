//! Conversation memory: Rig-managed persistent conversation history for agents.
//!
//! Memory differs from existing agent context features:
//! - classic runtime context: static documents always included in prompts;
//! - classic runtime request patches: per-turn documents supplied by application hooks;
//! - caller-managed message history supplied directly on completion requests;
//! - **Memory** (this module): Rig-managed history loaded and saved automatically per
//!   conversation id.
//!
//! # Example
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
    Backend(MemoryBackendError),

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

/// Owned-future companions to [`ConversationMemory`]: the same operations,
/// but taking an `Arc`'d backend and an owned [`ConversationId`] so the
/// returned future is `'static` — the shape an executor-agnostic host (an ECS
/// system spawning a task, a channel pump) needs, without per-call-site
/// clone-into-`async move` ceremony. The borrowed trait methods remain the
/// implementation surface; these are wrappers.
pub trait ConversationMemoryExt: ConversationMemory {
    /// [`ConversationMemory::load`] as an owned, `'static` future.
    fn load_owned(
        self: Arc<Self>,
        conversation_id: ConversationId,
    ) -> WasmBoxedFuture<'static, Result<Vec<Message>, MemoryError>>
    where
        Self: 'static,
    {
        Box::pin(async move { self.load(&conversation_id).await })
    }

    /// [`ConversationMemory::append`] as an owned, `'static` future.
    fn append_owned(
        self: Arc<Self>,
        conversation_id: ConversationId,
        messages: Vec<Message>,
    ) -> WasmBoxedFuture<'static, Result<(), MemoryError>>
    where
        Self: 'static,
    {
        Box::pin(async move { self.append(&conversation_id, messages).await })
    }

    /// [`ConversationMemory::clear`] as an owned, `'static` future.
    fn clear_owned(
        self: Arc<Self>,
        conversation_id: ConversationId,
    ) -> WasmBoxedFuture<'static, Result<(), MemoryError>>
    where
        Self: 'static,
    {
        Box::pin(async move { self.clear(&conversation_id).await })
    }
}

impl<M: ConversationMemory + ?Sized> ConversationMemoryExt for M {}

// Forwarding impls so callers can pass smart pointers (`Arc<M>`, `Box<M>`,
// including unsized trait objects) wherever a memory trait is expected. Each
// arm forwards every method of one trait through `(**self)` for the listed
// pointer types.
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

/// A side-channel for messages that a memory policy or adapter removes from
/// active history during [`ConversationMemory::load`].
///
/// Truncating policies (sliding window, token budget, …) drop older turns
/// once their limit is exceeded. Without a hook those messages are silently
/// lost. A [`DemotionHook`] receives the demoted messages and can persist
/// them into a long-tail store (semantic memory, episodic recall, archival
/// storage, …), turning truncation into demotion.
///
/// The trait is defined here in `rig-core` so that *any* memory backend
/// (in-memory, vector store, file archive, …) can implement it without
/// taking on a `rig-memory` dependency. The composing adapter that actually
/// wires a [`ConversationMemory`] backend, a policy, and a hook together
/// lives in the `rig-memory` companion crate.
///
/// Hooks should be inexpensive: their future is awaited inline on every
/// `load` that produces demoted messages, so a slow hook delays the agent's
/// next turn. Offload heavy I/O (network writes, disk fsyncs, …) to a
/// background task or a buffered channel inside the implementation.
///
/// # Idempotency contract
///
/// Implementations **must** be idempotent on the
/// `(conversation_id, messages)` pair. Composing adapters such as the
/// `DemotingPolicyMemory` in `rig-memory` track in-process delivery
/// watermarks to avoid replaying the same demotion within a single
/// process lifetime, but those watermarks are not persisted: across
/// process restarts (or when a new adapter is constructed over an
/// existing backend) the hook will receive previously-delivered
/// messages again. Hooks that append to durable storage should
/// deduplicate by content hash, by `(conversation_id, message_id)`,
/// or by an equivalent stable key.
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

// Forwarding impl so callers can pass `Arc<H>` wherever a `DemotionHook`
// is expected (e.g. when sharing a single hook between multiple memory
// adapters).
forward_memory_trait!(DemotionHook: Arc);

/// Derives a single [`Message`]-shaped artifact from a slice of messages
/// that a memory policy has evicted from the active window.
///
/// Where a [`DemotionHook`] is a one-way drain — observe what fell out and
/// return `()` — a `Compactor` is the inverse: it takes the evicted prefix
/// (and optionally the previous summary) and produces a derived artifact
/// that the composing adapter splices *back into* the active history. The
/// resulting prompt is no longer a verbatim suffix of the conversation; it
/// is `[summary, ...recent_window]`.
///
/// Implementations typically wrap an LLM call (`LlmCompactor<M>`) or a
/// pure template rollup. They run inline on the load path whenever the
/// policy demotes new messages, so a slow compactor delays the agent's
/// next turn — keep them fast or offload to a cached/background pipeline.
///
/// # Rolling summaries
///
/// `carry_over` is the artifact produced by the previous compaction for
/// this conversation, if any. Implementations that want a *recursive*
/// summary (the canonical pattern for long-running agents) should
/// summarize `evicted` *together with* `carry_over` so context lost in
/// earlier compactions is preserved transitively. Stateless implementations
/// can ignore `carry_over` and produce a fresh summary of `evicted` alone.
///
/// # Idempotency contract
///
/// Composing adapters track per-conversation in-process delivery so the
/// same `evicted` slice is not compacted twice within a process lifetime,
/// but those watermarks are not persisted across restarts. Implementations
/// that have side effects (writing summaries to a vector store, billing an
/// LLM call) should deduplicate by conversation id and content hash, the
/// same way [`DemotionHook`] implementations do.
pub trait Compactor: WasmCompatSend + WasmCompatSync {
    /// The summary value produced by [`Compactor::compact`].
    ///
    /// `Into<Message>` is required so the composing adapter can splice the
    /// artifact at the front of the loaded history. `Clone` is required so
    /// the adapter can keep a private copy as `carry_over` for the next
    /// compaction.
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

// Forwarding impl so callers can pass `Arc<C>` wherever a `Compactor` is
// expected (e.g. when sharing a single compactor across adapters).
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
mod tests {
    use super::*;
    use crate::completion::Message;

    fn user(text: &str) -> Message {
        Message::user(text)
    }

    fn assistant(text: &str) -> Message {
        Message::assistant(text)
    }

    /// The `*_owned` companions return `Send + 'static` futures (spawnable on
    /// any executor) and behave exactly like the borrowed methods, including
    /// through an `Arc<dyn ConversationMemory>`.
    #[tokio::test]
    async fn owned_futures_are_static_and_match_borrowed_behavior() {
        fn assert_send_static<T: Send + 'static>(value: T) -> T {
            value
        }

        let mem: Arc<dyn ConversationMemory> = Arc::new(InMemoryConversationMemory::new());
        let id = ConversationId::from("c-owned");
        assert_send_static(mem.clone().append_owned(id.clone(), vec![user("hello")]))
            .await
            .unwrap();
        let loaded = assert_send_static(mem.clone().load_owned(id.clone()))
            .await
            .unwrap();
        assert_eq!(loaded.len(), 1);
        assert_send_static(mem.clone().clear_owned(id.clone()))
            .await
            .unwrap();
        assert!(mem.load(&id).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn round_trip() {
        let mem = InMemoryConversationMemory::new();
        assert!(mem.load(&"c1".into()).await.unwrap().is_empty());

        mem.append(&"c1".into(), vec![user("hello"), assistant("hi")])
            .await
            .unwrap();

        let loaded = mem.load(&"c1".into()).await.unwrap();
        assert_eq!(loaded.len(), 2);
    }

    #[tokio::test]
    async fn isolation_between_conversations() {
        let mem = InMemoryConversationMemory::new();
        mem.append(&"a".into(), vec![user("hi a")]).await.unwrap();
        mem.append(&"b".into(), vec![user("hi b")]).await.unwrap();

        assert_eq!(mem.load(&"a".into()).await.unwrap().len(), 1);
        assert_eq!(mem.load(&"b".into()).await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn clear_removes_history() {
        let mem = InMemoryConversationMemory::new();
        mem.append(&"c".into(), vec![user("x")]).await.unwrap();
        mem.clear(&"c".into()).await.unwrap();
        assert!(mem.load(&"c".into()).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn with_filter_transforms_loaded_messages() {
        let mem = InMemoryConversationMemory::new()
            .with_filter(|msgs: Vec<Message>| msgs.into_iter().rev().take(2).collect());

        mem.append(
            &"c".into(),
            vec![user("1"), assistant("2"), user("3"), assistant("4")],
        )
        .await
        .unwrap();

        let loaded = mem.load(&"c".into()).await.unwrap();
        assert_eq!(loaded.len(), 2, "filter should retain only 2 messages");
    }

    #[tokio::test]
    async fn arc_conversation_memory_forwards_to_inner() {
        let inner = Arc::new(InMemoryConversationMemory::new());
        let mem: Arc<dyn ConversationMemory> = inner.clone();

        mem.append(&"c".into(), vec![user("hello")]).await.unwrap();

        assert_eq!(inner.load(&"c".into()).await.unwrap().len(), 1);
        mem.clear(&"c".into()).await.unwrap();
        assert!(inner.load(&"c".into()).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn boxed_conversation_memory_forwards_to_inner() {
        let mem: Box<dyn ConversationMemory> = Box::new(InMemoryConversationMemory::new());

        mem.append(&"c".into(), vec![user("hello")]).await.unwrap();

        assert_eq!(mem.load(&"c".into()).await.unwrap().len(), 1);
        mem.clear(&"c".into()).await.unwrap();
        assert!(mem.load(&"c".into()).await.unwrap().is_empty());
    }
}
