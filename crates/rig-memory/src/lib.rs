#![cfg_attr(
    test,
    allow(
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::unwrap_used,
        clippy::unreachable
    )
)]
//! Conversation memory policies for the Rig agent framework.
//!
//! `rig-core` provides the [`ConversationMemory`] trait and an in-process
//! [`InMemoryConversationMemory`] backend. This crate adds reusable, named
//! transformations for shaping loaded history before it is sent to the model:
//!
//! - [`NoopMemoryPolicy`] — identity, returns input unchanged.
//! - [`SlidingWindowMemory`] — retains the most recent `N` messages.
//! - [`TokenWindowMemory`] — retains messages that fit within a token budget.
//! - [`HeuristicTokenCounter`] — provider-agnostic, zero-dependency
//!   [`TokenCounter`] that approximates token cost from character lengths.
//! - [`DemotionHook`] + [`DemotingPolicyMemory`] — bridge truncated turns
//!   from a [`MemoryPolicy`] into a long-tail store.
//! - [`Compactor`] + [`CompactingMemory`] — replace truncated turns with a
//!   derived summary artifact (rolling-summary semantics).
//! - [`TemplateCompactor`] — zero-dependency reference [`Compactor`] that
//!   produces a textual rollup without calling an LLM.
//!
//! All sliding policies drop a leading orphan tool-result message when the
//! preceding assistant tool call has been truncated, since most providers
//! reject unpaired tool results.
//!
//! # Example
//!
//! ```
//! use rig_memory::{InMemoryConversationMemory, IntoFilter, SlidingWindowMemory};
//!
//! let memory = InMemoryConversationMemory::new()
//!     .with_filter(SlidingWindowMemory::last_messages(20).into_filter());
//! ```

use std::{
    borrow::Cow,
    collections::HashMap,
    sync::{Arc, Mutex as StdMutex},
};

/// Re-exports of the core memory abstractions so callers only need a single
/// dependency on `rig-memory` for both the trait/backend and the policies.
pub use rig_core::memory::{
    Compactor, ConversationMemory, DemotionHook, InMemoryConversationMemory, MemoryError,
    NoopDemotionHook,
};

use rig_core::completion::Message;
use rig_core::id::ConversationId;
use rig_core::message::UserContent;
use rig_core::wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync};

/// A transformation applied to messages loaded from a [`ConversationMemory`].
///
/// Policies typically truncate, summarize, or re-order history. They are
/// pure, fallible message transformers: implementors that cannot fail should
/// always return `Ok`.
pub trait MemoryPolicy: WasmCompatSend + WasmCompatSync {
    /// Transform `messages` into the history that should be returned to the
    /// agent. This is the required method — every policy must implement it.
    fn apply(&self, messages: Vec<Message>) -> Result<Vec<Message>, MemoryError>;

    /// Transform `messages` and report which messages were demoted (excluded
    /// from the returned history).
    ///
    /// Returns `(kept, demoted)`. The default implementation returns
    /// `(self.apply(messages)?, Vec::new())`, which is correct for
    /// non-truncating policies. Truncating policies (sliding window, token
    /// window, …) override this method to populate `demoted` with the
    /// messages they evicted.
    ///
    /// Implementors must guarantee that `demoted` is the prefix of the
    /// original input not retained in `kept`, in original order. Composing
    /// adapters such as [`DemotingPolicyMemory`] rely on this contract to
    /// track delivery watermarks correctly.
    fn apply_with_demoted(
        &self,
        messages: Vec<Message>,
    ) -> Result<(Vec<Message>, Vec<Message>), MemoryError> {
        Ok((self.apply(messages)?, Vec::new()))
    }
}

impl<P> MemoryPolicy for Arc<P>
where
    P: MemoryPolicy + ?Sized,
{
    fn apply(&self, messages: Vec<Message>) -> Result<Vec<Message>, MemoryError> {
        (**self).apply(messages)
    }

    fn apply_with_demoted(
        &self,
        messages: Vec<Message>,
    ) -> Result<(Vec<Message>, Vec<Message>), MemoryError> {
        (**self).apply_with_demoted(messages)
    }
}

impl<P> MemoryPolicy for Box<P>
where
    P: MemoryPolicy + ?Sized,
{
    fn apply(&self, messages: Vec<Message>) -> Result<Vec<Message>, MemoryError> {
        (**self).apply(messages)
    }

    fn apply_with_demoted(
        &self,
        messages: Vec<Message>,
    ) -> Result<(Vec<Message>, Vec<Message>), MemoryError> {
        (**self).apply_with_demoted(messages)
    }
}

/// Adapt a [`MemoryPolicy`] into a closure suitable for
/// [`InMemoryConversationMemory::with_filter`].
///
/// Errors raised by the policy are swallowed because `with_filter` does not
/// propagate failures. Use [`MemoryPolicy::apply`] directly when you need to
/// observe policy errors.
pub trait IntoFilter: MemoryPolicy + Sized + 'static {
    /// Convert this policy into a filter closure.
    ///
    /// On policy error the original input is returned unchanged and a
    /// `tracing::warn!` is emitted, so a transient policy bug degrades
    /// gracefully (the model still sees the unfiltered history) instead of
    /// silently erasing context.
    fn into_filter(self) -> BoxedFilter {
        let policy = Arc::new(self);
        Box::new(move |msgs| {
            // Deliberate clone: `apply` consumes the history, and the
            // graceful-degradation contract above requires handing the
            // original back when the policy errors.
            let fallback = msgs.clone();
            match policy.apply(msgs) {
                Ok(out) => out,
                Err(err) => {
                    tracing::warn!(error = %err, "memory policy failed; returning unfiltered history");
                    fallback
                }
            }
        })
    }
}

/// Boxed filter closure returned by [`IntoFilter::into_filter`].
#[cfg(not(target_family = "wasm"))]
pub type BoxedFilter = Box<dyn Fn(Vec<Message>) -> Vec<Message> + Send + Sync>;

/// Boxed filter closure returned by [`IntoFilter::into_filter`].
#[cfg(target_family = "wasm")]
pub type BoxedFilter = Box<dyn Fn(Vec<Message>) -> Vec<Message>>;

impl<P> IntoFilter for P where P: MemoryPolicy + 'static {}

/// A [`MemoryPolicy`] that returns its input unchanged.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopMemoryPolicy;

impl MemoryPolicy for NoopMemoryPolicy {
    fn apply(&self, messages: Vec<Message>) -> Result<Vec<Message>, MemoryError> {
        Ok(messages)
    }
}

/// A [`MemoryPolicy`] that retains only the most recent `max_messages` entries.
///
/// When the window starts mid-conversation, a leading orphan tool-result
/// message (a [`Message::User`] whose first content is a tool result without
/// its preceding [`Message::Assistant`] tool call) is dropped to preserve the
/// tool-call/result pairing required by most providers.
#[derive(Debug, Clone, Copy)]
pub struct SlidingWindowMemory {
    max_messages: usize,
}

impl SlidingWindowMemory {
    /// Keep at most `n` messages.
    pub fn last_messages(n: usize) -> Self {
        Self { max_messages: n }
    }
}

impl MemoryPolicy for SlidingWindowMemory {
    fn apply(&self, messages: Vec<Message>) -> Result<Vec<Message>, MemoryError> {
        Ok(self.apply_with_demoted(messages)?.0)
    }

    fn apply_with_demoted(
        &self,
        messages: Vec<Message>,
    ) -> Result<(Vec<Message>, Vec<Message>), MemoryError> {
        if messages.len() <= self.max_messages {
            return Ok((messages, Vec::new()));
        }

        let keep_from = messages.len() - self.max_messages;
        Ok(split_window(messages, keep_from))
    }
}

/// Split `messages` at `keep_from` into `(window, demoted)`.
///
/// A window that opens on a tool result has lost the assistant call it
/// answers, which providers reject; that orphan joins the demoted set rather
/// than being dropped, so the demotion hook still observes it end-to-end even
/// though the model never sees it again.
fn split_window(messages: Vec<Message>, keep_from: usize) -> (Vec<Message>, Vec<Message>) {
    let mut iter = messages.into_iter();
    let mut demoted: Vec<Message> = (&mut iter).take(keep_from).collect();
    let mut window: Vec<Message> = iter.collect();

    if let Some(Message::User { content }) = window.first()
        && matches!(content.first(), Some(UserContent::ToolResult(_)))
    {
        demoted.push(window.remove(0));
    }

    (window, demoted)
}

/// Counts the tokens contributed by a single [`Message`].
///
/// Implementors should pick a counting strategy appropriate for their target
/// provider (for example, `tiktoken-rs` for OpenAI). Counting must be cheap;
/// it runs once per message on every memory load.
pub trait TokenCounter: WasmCompatSend + WasmCompatSync {
    /// Approximate the number of tokens contributed by `message`.
    fn count(&self, message: &Message) -> usize;
}

impl<F> TokenCounter for F
where
    F: Fn(&Message) -> usize + WasmCompatSend + WasmCompatSync,
{
    fn count(&self, message: &Message) -> usize {
        (self)(message)
    }
}

impl<C> TokenCounter for Arc<C>
where
    C: TokenCounter + ?Sized,
{
    fn count(&self, message: &Message) -> usize {
        (**self).count(message)
    }
}

impl TokenCounter for Box<dyn TokenCounter> {
    fn count(&self, message: &Message) -> usize {
        (**self).count(message)
    }
}

/// A provider-agnostic [`TokenCounter`] that approximates token counts from
/// UTF-8 byte lengths.
///
/// This is intended as a zero-dependency default. It is **not** a substitute
/// for a tokenizer and will under- or over-count by up to ~30 % on real
/// content, but it is monotonic in message size and stable across runs, which
/// is enough for [`TokenWindowMemory`] to enforce a budget that *trends*
/// with provider billing.
///
/// # Strategy
///
/// For every text-bearing block (`Text`, reasoning text, tool-result text)
/// the counter sums UTF-8 byte lengths (`str::len`, an O(1) call) and divides
/// by `bytes_per_token`, rounded up. Bytes are used instead of Unicode
/// scalars because the cost is O(1), modern BPE tokenizers operate on byte
/// sequences, and per-message budgeting only needs the rough order of
/// magnitude. For ASCII text bytes and characters coincide; for non-ASCII
/// text the counter slightly over-estimates, which is the safe direction
/// for a hard budget.
///
/// Tool calls are charged the JSON-serialised length of their `ToolFunction`
/// payload. Each message is charged a flat `per_message_overhead` to model
/// the per-turn role/separator tokens that providers add internally. Non-text
/// blocks (images, audio, video, documents) are charged
/// `per_attachment_tokens` each because their real cost is provider-specific
/// and rarely text-derived.
///
/// # Presets
///
/// The defaults match OpenAI's published rule of thumb (~4 bytes per token,
/// ~4 tokens of per-message overhead). [`HeuristicTokenCounter::anthropic`]
/// uses a slightly denser ratio that better fits Claude's tokenizer.
///
/// # Example
///
/// ```
/// use rig_memory::{HeuristicTokenCounter, TokenWindowMemory};
///
/// let policy = TokenWindowMemory::new(2_000, HeuristicTokenCounter::default());
/// # let _ = policy;
/// ```
#[derive(Debug, Clone, Copy)]
pub struct HeuristicTokenCounter {
    bytes_per_token: f32,
    per_message_overhead: usize,
    per_attachment_tokens: usize,
}

impl HeuristicTokenCounter {
    /// Create a counter with explicit parameters.
    ///
    /// `bytes_per_token` is clamped to a minimum of `1.0` so the counter
    /// never panics or produces zero-cost messages on degenerate input.
    pub fn new(
        bytes_per_token: f32,
        per_message_overhead: usize,
        per_attachment_tokens: usize,
    ) -> Self {
        let bytes_per_token = if bytes_per_token.is_finite() && bytes_per_token >= 1.0 {
            bytes_per_token
        } else {
            1.0
        };
        Self {
            bytes_per_token,
            per_message_overhead,
            per_attachment_tokens,
        }
    }

    /// Preset matching OpenAI's chat-completion token rule of thumb.
    ///
    /// Equivalent to [`HeuristicTokenCounter::default`].
    pub fn openai() -> Self {
        Self::new(4.0, 4, 256)
    }

    /// Preset tuned for Anthropic Claude's tokenizer.
    pub fn anthropic() -> Self {
        Self::new(3.5, 4, 256)
    }

    /// Preset tuned for Google Gemini.
    pub fn gemini() -> Self {
        Self::new(4.0, 4, 256)
    }

    fn bytes_to_tokens(&self, bytes: usize) -> usize {
        // `bytes_per_token` is clamped to >= 1.0 in the constructor, so the
        // division is well-defined. We round up so a single non-empty
        // input still costs at least one token.
        let tokens = (bytes as f32) / self.bytes_per_token;
        tokens.ceil() as usize
    }

    fn count_user(&self, content: &rig_core::message::UserContent) -> usize {
        use rig_core::message::UserContent;
        match content {
            UserContent::Text(text) => self.bytes_to_tokens(text.text.len()),
            UserContent::ToolResult(result) => result
                .content
                .iter()
                .map(|c| match c {
                    rig_core::message::ToolResultContent::Text(t) => {
                        self.bytes_to_tokens(t.text.len())
                    }
                    rig_core::message::ToolResultContent::Json { value } => {
                        self.bytes_to_tokens(value.to_string().len())
                    }
                    rig_core::message::ToolResultContent::Image(_) => self.per_attachment_tokens,
                })
                .sum(),
            UserContent::Image(_)
            | UserContent::Audio(_)
            | UserContent::Video(_)
            | UserContent::Document(_) => self.per_attachment_tokens,
        }
    }

    fn count_assistant(&self, content: &rig_core::message::AssistantContent) -> usize {
        use rig_core::message::AssistantContent;
        match content {
            AssistantContent::Text(text) => self.bytes_to_tokens(text.text.len()),
            AssistantContent::Reasoning(reasoning) => {
                self.bytes_to_tokens(reasoning.display_text().len())
            }
            AssistantContent::ToolCall(call) => {
                let name_bytes = call.function.name.len();
                // `serde_json::Value::to_string` is the canonical compact JSON
                // encoding and never fails, so we charge tool calls by the
                // length of their serialised arguments without pulling in a
                // direct `serde_json` dependency.
                let args_bytes = call.function.arguments.to_string().len();
                self.bytes_to_tokens(name_bytes + args_bytes)
            }
            AssistantContent::Image(_) => self.per_attachment_tokens,
        }
    }
}

impl Default for HeuristicTokenCounter {
    fn default() -> Self {
        Self::openai()
    }
}

impl TokenCounter for HeuristicTokenCounter {
    fn count(&self, message: &Message) -> usize {
        let content_tokens: usize = match message {
            Message::User { content } => content.iter().map(|c| self.count_user(c)).sum(),
            Message::Assistant { content, .. } => {
                content.iter().map(|c| self.count_assistant(c)).sum()
            }
            Message::System { content } => self.bytes_to_tokens(content.len()),
        };
        content_tokens.saturating_add(self.per_message_overhead)
    }
}

/// A [`MemoryPolicy`] that retains the most recent messages up to a token budget.
///
/// Messages are walked from newest to oldest, accumulating token counts
/// produced by a [`TokenCounter`]. Once including a message would exceed
/// `max_tokens`, the walk stops and the included messages are returned in
/// original (oldest-first) order. As with [`SlidingWindowMemory`], a leading
/// orphan tool-result is dropped when its paired assistant tool call has
/// been truncated.
pub struct TokenWindowMemory {
    max_tokens: usize,
    counter: Arc<dyn TokenCounter>,
}

impl TokenWindowMemory {
    /// Create a new policy with a token budget and a counter.
    pub fn new<C>(max_tokens: usize, counter: C) -> Self
    where
        C: TokenCounter + 'static,
    {
        Self {
            max_tokens,
            counter: Arc::new(counter),
        }
    }
}

impl std::fmt::Debug for TokenWindowMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenWindowMemory")
            .field("max_tokens", &self.max_tokens)
            .field("counter", &"<counter>")
            .finish()
    }
}

impl MemoryPolicy for TokenWindowMemory {
    fn apply(&self, messages: Vec<Message>) -> Result<Vec<Message>, MemoryError> {
        Ok(self.apply_with_demoted(messages)?.0)
    }

    fn apply_with_demoted(
        &self,
        messages: Vec<Message>,
    ) -> Result<(Vec<Message>, Vec<Message>), MemoryError> {
        let mut budget = self.max_tokens;
        let mut keep_from = messages.len();

        for (idx, msg) in messages.iter().enumerate().rev() {
            let cost = self.counter.count(msg);
            if cost > budget {
                break;
            }
            budget -= cost;
            keep_from = idx;
        }

        Ok(split_window(messages, keep_from))
    }
}

/// Wrap a [`ConversationMemory`] backend with a [`MemoryPolicy`], propagating
/// policy errors to the caller as [`MemoryError::Policy`].
///
/// This is the hard-fail counterpart to
/// [`InMemoryConversationMemory::with_filter`] + [`IntoFilter::into_filter`].
/// `with_filter` swallows policy errors and returns the unfiltered history;
/// `PolicyMemory` surfaces them so callers can decide how to react.
///
/// # Example
///
/// ```no_run
/// use rig_memory::{InMemoryConversationMemory, PolicyMemory, SlidingWindowMemory};
///
/// let memory = PolicyMemory::new(
///     InMemoryConversationMemory::new(),
///     SlidingWindowMemory::last_messages(20),
/// );
/// ```
#[derive(Debug, Clone, Copy)]
pub struct PolicyMemory<M, P> {
    inner: M,
    policy: P,
}

impl<M, P> PolicyMemory<M, P> {
    /// Wrap `inner` so every loaded history is run through `policy`.
    pub fn new(inner: M, policy: P) -> Self {
        Self { inner, policy }
    }

    /// Return a reference to the wrapped backend.
    pub fn inner(&self) -> &M {
        &self.inner
    }

    /// Return a reference to the wrapped policy.
    pub fn policy(&self) -> &P {
        &self.policy
    }

    /// Consume the wrapper and return the underlying backend and policy.
    pub fn into_inner(self) -> (M, P) {
        (self.inner, self.policy)
    }
}

impl<M, P> ConversationMemory for PolicyMemory<M, P>
where
    M: ConversationMemory,
    P: MemoryPolicy,
{
    fn load<'a>(
        &'a self,
        conversation_id: &'a ConversationId,
    ) -> WasmBoxedFuture<'a, Result<Vec<Message>, MemoryError>> {
        Box::pin(async move {
            let messages = self.inner.load(conversation_id).await?;
            self.policy.apply(messages)
        })
    }

    fn append<'a>(
        &'a self,
        conversation_id: &'a ConversationId,
        messages: Vec<Message>,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
        self.inner.append(conversation_id, messages)
    }

    fn clear<'a>(
        &'a self,
        conversation_id: &'a ConversationId,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
        self.inner.clear(conversation_id)
    }
}

/// A [`ConversationMemory`] adapter that wraps a backend with a
/// [`MemoryPolicy`] **and** a [`DemotionHook`], so messages truncated by the
/// policy flow into the hook before the active window is returned.
///
/// `DemotingPolicyMemory` is the bridge between the recent-turn store
/// ([`InMemoryConversationMemory`] or any other [`ConversationMemory`]) and a
/// long-tail store (`MemvidPersistHook`, vector RAG, archival storage, …).
/// Compose it with any [`MemoryPolicy`] that overrides
/// [`MemoryPolicy::apply_with_demoted`]; policies that rely on the default
/// implementation will still load correctly but will never demote anything.
///
/// # Concurrency
///
/// Concurrent [`ConversationMemory::load`] calls on the same
/// `conversation_id` are serialised at the demotion seam: only one call at
/// a time delivers messages to the hook for a given conversation. Other
/// concurrent loads for that conversation observe the in-flight delivery
/// and return the truncated `kept` history immediately without firing the
/// hook again. Pending demotions that were skipped this way are picked up
/// by the next `load` after the in-flight delivery completes.
///
/// **Failure visibility.** A hook error is returned only to the caller
/// whose `load` actually drove the delivery. Concurrent callers that
/// short-circuited on `in_flight` see `Ok(kept)` even if the in-flight
/// delivery ultimately failed; the watermark stays unchanged so the next
/// `load` retries. Callers that rely on the hook for durability should
/// treat a successful `load` as best-effort with respect to demotion and
/// surface hook failures through the hook's own observability (logs,
/// metrics, dead-letter buffer) rather than the `load` return value.
///
/// # Persistence
///
/// Delivery watermarks are kept in process memory only. Across process
/// restarts, the hook will receive previously-delivered demotions again;
/// see the [`DemotionHook`] idempotency contract.
///
/// # Example
///
/// ```no_run
/// use rig_memory::{
///     DemotingPolicyMemory, DemotionHook, InMemoryConversationMemory,
///     MemoryError, NoopDemotionHook, SlidingWindowMemory,
/// };
///
/// let memory = DemotingPolicyMemory::new(
///     InMemoryConversationMemory::new(),
///     SlidingWindowMemory::last_messages(20),
///     NoopDemotionHook,
/// );
/// # let _ = memory;
/// ```
pub struct DemotingPolicyMemory<M, P, H> {
    inner: M,
    policy: P,
    hook: H,
    state: StdMutex<HashMap<ConversationId, ConversationDemotionState>>,
}

type InFlightReservation = Arc<()>;

/// Emits the members shared by the stateful wrappers
/// ([`DemotingPolicyMemory`], [`CompactingMemory`]): accessors,
/// `forget`/`tracked_conversations` over the per-conversation state map, and
/// a `Debug` impl that elides the non-`Debug` third component.
macro_rules! stateful_wrapper_common {
    ($ty:ident, $third:ident: $tgen:ident $(: $tbound:ident)?) => {
        impl<M, P, $tgen $(: $tbound)?> $ty<M, P, $tgen> {
            /// Return a reference to the wrapped backend.
            pub fn inner(&self) -> &M {
                &self.inner
            }

            /// Return a reference to the wrapped policy.
            pub fn policy(&self) -> &P {
                &self.policy
            }

            /// Return a reference to the third component.
            pub fn $third(&self) -> &$tgen {
                &self.$third
            }

            /// Consume the wrapper and return its three components.
            pub fn into_inner(self) -> (M, P, $tgen) {
                (self.inner, self.policy, self.$third)
            }

            /// Drop the in-process state for `conversation_id`.
            ///
            /// Call this when a conversation has ended to bound memory usage;
            /// the state map is otherwise unbounded — entries persist for the
            /// lifetime of the wrapper. If the internal state lock has been
            /// poisoned by a panic in another thread, this is a no-op (the
            /// state will be dropped naturally when the wrapper itself is
            /// dropped).
            pub fn forget(&self, conversation_id: &ConversationId) {
                if let Ok(mut guard) = self.state.lock() {
                    guard.remove(conversation_id);
                }
            }

            /// Number of conversations currently tracked in the state map.
            /// Useful for telemetry and leak detection. Returns `0` if the
            /// internal state lock is poisoned.
            pub fn tracked_conversations(&self) -> usize {
                self.state.lock().map_or(0, |g| g.len())
            }
        }

        impl<M, P, $tgen $(: $tbound)?> std::fmt::Debug for $ty<M, P, $tgen>
        where
            M: std::fmt::Debug,
            P: std::fmt::Debug,
        {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct(stringify!($ty))
                    .field("inner", &self.inner)
                    .field("policy", &self.policy)
                    .field(
                        stringify!($third),
                        &concat!("<", stringify!($third), ">"),
                    )
                    .finish()
            }
        }
    };
}

/// Emits the delegating `append` and the `clear`-then-`forget` methods of a
/// stateful wrapper's [`ConversationMemory`] impl.
macro_rules! stateful_wrapper_append_clear {
    () => {
        fn append<'a>(
            &'a self,
            conversation_id: &'a ConversationId,
            messages: Vec<Message>,
        ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
            self.inner.append(conversation_id, messages)
        }

        fn clear<'a>(
            &'a self,
            conversation_id: &'a ConversationId,
        ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
            Box::pin(async move {
                self.inner.clear(conversation_id).await?;
                self.forget(conversation_id);
                Ok(())
            })
        }
    };
}

#[derive(Debug, Default, Clone)]
struct ConversationDemotionState {
    /// Number of demoted messages already delivered to the hook within
    /// this process lifetime. Advanced only on hook success.
    delivered: usize,
    /// Reservation held while a `load` is currently awaiting
    /// `hook.on_demote(...)` for this conversation. Other concurrent loads
    /// observe this and short-circuit without re-delivering the same messages.
    in_flight: Option<InFlightReservation>,
}

impl<M, P, H> DemotingPolicyMemory<M, P, H> {
    /// Wrap `inner` so every load runs through `policy` and demoted messages
    /// flow into `hook`.
    pub fn new(inner: M, policy: P, hook: H) -> Self {
        Self {
            inner,
            policy,
            hook,
            state: StdMutex::new(HashMap::new()),
        }
    }
}

stateful_wrapper_common!(DemotingPolicyMemory, hook: H);

impl<M, P, H> ConversationMemory for DemotingPolicyMemory<M, P, H>
where
    M: ConversationMemory,
    P: MemoryPolicy,
    H: DemotionHook,
{
    fn load<'a>(
        &'a self,
        conversation_id: &'a ConversationId,
    ) -> WasmBoxedFuture<'a, Result<Vec<Message>, MemoryError>> {
        Box::pin(async move {
            let messages = self.inner.load(conversation_id).await?;
            let (kept, mut demoted) = self.policy.apply_with_demoted(messages)?;
            let demoted_count = demoted.len();

            // Reserve a delivery slot atomically. Decide-and-mark must
            // happen under one short-lived lock so concurrent loads on
            // the same conversation_id can't both observe the same
            // delivered watermark and double-fire the hook.
            //
            // Fast path: if the conversation is already tracked, mutate in
            // place. Only allocate a new `String` key when we are about to
            // record state for a conversation we have not seen before *and*
            // there is actually demotion work to track.
            let (pending, reservation) = {
                let mut guard = self.state.lock().map_err(poisoned)?;
                if let Some(entry) = guard.get_mut(conversation_id) {
                    if entry.in_flight.is_some() {
                        // Another load is mid-delivery for this conversation;
                        // skip and let the next load see whatever it leaves
                        // behind.
                        return Ok(kept);
                    }
                    if entry.delivered >= demoted_count {
                        (Vec::new(), None)
                    } else {
                        let split = entry.delivered;
                        let reservation = Arc::new(());
                        entry.in_flight = Some(reservation.clone());
                        (demoted.split_off(split), Some(reservation))
                    }
                } else if demoted_count == 0 {
                    // First load for this conversation and nothing was
                    // demoted: no need to allocate a tracking entry yet.
                    (Vec::new(), None)
                } else {
                    let reservation = Arc::new(());
                    guard.insert(
                        conversation_id.clone(),
                        ConversationDemotionState {
                            delivered: 0,
                            in_flight: Some(reservation.clone()),
                        },
                    );
                    (std::mem::take(&mut demoted), Some(reservation))
                }
            };

            let Some(reservation) = reservation else {
                return Ok(kept);
            };

            // Arm an RAII guard so the in-flight gate is released even if
            // this future is dropped mid-await (caller cancellation) or the
            // hook panics. The reservation token prevents stale guards from
            // clearing newer in-flight loads after clear()/forget() reuse the
            // same conversation id.
            let in_flight_guard =
                InFlightGuard::new(&self.state, conversation_id, reservation.clone());

            let result = self.hook.on_demote(conversation_id, pending).await;

            // Reacquire briefly to advance the watermark on success and
            // always clear the in-flight flag so a future load can retry.
            //
            // Only update if the entry still exists: a concurrent `clear`
            // (and matching `forget`) for this `conversation_id` may have
            // dropped the watermark entry while the hook was awaiting. In
            // that case we must not resurrect it with a stale `delivered`
            // count — the next load on a freshly-populated backend would
            // then skip a real demotion.
            release_in_flight(&self.state, conversation_id, &reservation, |entry| {
                if result.is_ok() {
                    entry.delivered = demoted_count;
                }
            })?;
            in_flight_guard.disarm();
            result?;
            Ok(kept)
        })
    }

    stateful_wrapper_append_clear!();
}

fn poisoned<E: std::fmt::Display>(err: E) -> MemoryError {
    MemoryError::Internal(err.to_string())
}

/// Clear the conversation's `in_flight` reservation if the entry still exists
/// and still holds `reservation`, running `on_match` on the entry under the
/// lock. Returns `on_match`'s value only when the reservation matched — a
/// missing entry (concurrent `clear`) or a newer reservation is a no-op, so
/// stale releases can never resurrect or clobber newer state.
fn release_in_flight<S: InFlightSlot, T>(
    state: &StdMutex<HashMap<ConversationId, S>>,
    key: &ConversationId,
    reservation: &InFlightReservation,
    on_match: impl FnOnce(&mut S) -> T,
) -> Result<Option<T>, MemoryError> {
    let mut guard = state.lock().map_err(poisoned)?;
    let Some(entry) = guard.get_mut(key) else {
        return Ok(None);
    };
    if entry
        .in_flight_mut()
        .as_ref()
        .is_some_and(|current| Arc::ptr_eq(current, reservation))
    {
        *entry.in_flight_mut() = None;
        return Ok(Some(on_match(entry)));
    }
    Ok(None)
}

/// A per-conversation state entry with an `in_flight` delivery reservation,
/// letting [`InFlightGuard`] work over both the demotion and compaction
/// state maps.
trait InFlightSlot {
    fn in_flight_mut(&mut self) -> &mut Option<InFlightReservation>;
}

impl InFlightSlot for ConversationDemotionState {
    fn in_flight_mut(&mut self) -> &mut Option<InFlightReservation> {
        &mut self.in_flight
    }
}

impl<A> InFlightSlot for ConversationCompactionState<A> {
    fn in_flight_mut(&mut self) -> &mut Option<InFlightReservation> {
        &mut self.in_flight
    }
}

/// RAII guard that clears the `in_flight` flag for a conversation in the
/// shared demotion/compaction state map when dropped, unless the consumer
/// explicitly disarms it after a successful post-await update.
///
/// This prevents the in-flight gate from leaking when the awaiting
/// `load(...)` future is dropped (caller timeout, `tokio::select!`, etc.)
/// or when the hook/compactor panics: in either case `Drop` runs and
/// releases the gate so subsequent loads can retry. A missing entry is a
/// no-op, covering the case where a concurrent `clear` removed the
/// conversation while delivery was awaiting.
struct InFlightGuard<'a, S: InFlightSlot> {
    state: &'a StdMutex<HashMap<ConversationId, S>>,
    key: &'a ConversationId,
    reservation: InFlightReservation,
    armed: bool,
}

impl<'a, S: InFlightSlot> InFlightGuard<'a, S> {
    fn new(
        state: &'a StdMutex<HashMap<ConversationId, S>>,
        key: &'a ConversationId,
        reservation: InFlightReservation,
    ) -> Self {
        Self {
            state,
            key,
            reservation,
            armed: true,
        }
    }

    /// Disable the `Drop` clean-up. Call after the post-await state
    /// update has already cleared `in_flight` while holding the lock.
    fn disarm(mut self) {
        self.armed = false;
    }
}

impl<S: InFlightSlot> Drop for InFlightGuard<'_, S> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        // A poisoned lock is ignored, matching pre-guard behavior.
        let _ = release_in_flight(self.state, self.key, &self.reservation, |_| ());
    }
}

/// A [`ConversationMemory`] adapter that wraps a backend with a
/// [`MemoryPolicy`] **and** a [`Compactor`], replacing truncated turns with
/// a summary artifact spliced at the front of the loaded history.
///
/// `CompactingMemory` is the next layer above [`DemotingPolicyMemory`]: a
/// demotion hook only *observes* what the policy evicted, while a compactor
/// *substitutes* the evicted prefix with a derived [`Message`]. The loaded
/// history shape is therefore `[summary_message, ...kept_window]` whenever
/// any compaction has occurred for the conversation, and just `kept_window`
/// otherwise. The summary itself is recomputed (rolled forward) on every
/// load that produces newly-evicted messages, so older summaries are folded
/// into newer ones via the compactor's `carry_over` parameter.
///
/// # Concurrency
///
/// Concurrent [`ConversationMemory::load`] calls on the same
/// `conversation_id` are serialised at the compaction seam: only one call
/// at a time invokes the compactor for a given conversation. Other
/// concurrent loads observe the in-flight compaction and immediately
/// return the previously-stored summary spliced in front of `kept`,
/// without re-running the compactor. Newly-evicted messages skipped this
/// way are folded into the next compaction.
///
/// **Failure visibility.** A compactor error is returned only to the
/// caller whose `load` actually drove the compaction. Concurrent callers
/// that short-circuited on `in_flight` see `Ok([old_summary?, ...kept])`
/// even if the in-flight compaction ultimately failed; the watermark
/// stays unchanged so the next `load` retries.
///
/// # Persistence
///
/// The carry-over summary and delivery watermarks are kept in process
/// memory only. Across process restarts, the first load on each
/// conversation re-evicts and re-compacts the same prefix; compactors
/// that have side effects (LLM calls, persistent writes) should
/// deduplicate.
///
/// # Prompt shape and budgets
///
/// `CompactingMemory` is **policy-agnostic**: the wrapped
/// [`MemoryPolicy`] decides which messages are kept versus demoted, and
/// only the kept window is bounded by that policy. The summary artifact
/// produced by the [`Compactor`] is spliced **outside** that budget — so
/// the loaded prompt has shape `[summary, ...kept_window]` where
/// `kept_window` respects the policy's bounds and `summary` adds an
/// extra message on top of it.
///
/// Callers that combine `CompactingMemory` with a token-budgeted policy
/// (e.g. [`TokenWindowMemory`]) **must use a [`Compactor`] that bounds
/// its own artifact**, or accept that the loaded prompt may exceed the
/// policy's budget by the size of the summary. The reference
/// [`TemplateCompactor`] grows monotonically by default; configure it
/// with [`TemplateCompactor::with_max_bytes`] to cap the rolled-up text.
///
/// # Example
///
/// ```no_run
/// use rig_memory::{
///     CompactingMemory, InMemoryConversationMemory, SlidingWindowMemory,
///     TemplateCompactor,
/// };
///
/// let memory = CompactingMemory::new(
///     InMemoryConversationMemory::new(),
///     SlidingWindowMemory::last_messages(20),
///     TemplateCompactor::new(),
/// );
/// # let _ = memory;
/// ```
pub struct CompactingMemory<M, P, C: Compactor> {
    inner: M,
    policy: P,
    compactor: C,
    state: StdMutex<HashMap<ConversationId, ConversationCompactionState<C::Artifact>>>,
}

struct ConversationCompactionState<A> {
    /// Latest summary artifact for this conversation, if compaction has
    /// already happened. Cloned into the loaded history on every `load`.
    summary: Option<A>,
    /// Number of demoted messages already absorbed into `summary` within
    /// this process lifetime. Advanced only on compactor success.
    absorbed: usize,
    /// Reservation held while a `load` is currently awaiting the compactor for
    /// this conversation. Other concurrent loads observe this and short-circuit
    /// without re-running the compactor.
    in_flight: Option<InFlightReservation>,
}

impl<M, P, C: Compactor> CompactingMemory<M, P, C> {
    /// Wrap `inner` so every load runs through `policy` and demoted messages
    /// are summarised by `compactor`.
    pub fn new(inner: M, policy: P, compactor: C) -> Self {
        Self {
            inner,
            policy,
            compactor,
            state: StdMutex::new(HashMap::new()),
        }
    }
}

stateful_wrapper_common!(CompactingMemory, compactor: C: Compactor);

impl<M, P, C> ConversationMemory for CompactingMemory<M, P, C>
where
    M: ConversationMemory,
    P: MemoryPolicy,
    C: Compactor,
{
    fn load<'a>(
        &'a self,
        conversation_id: &'a ConversationId,
    ) -> WasmBoxedFuture<'a, Result<Vec<Message>, MemoryError>> {
        Box::pin(async move {
            let messages = self.inner.load(conversation_id).await?;
            let (kept, demoted) = self.policy.apply_with_demoted(messages)?;
            let demoted_count = demoted.len();

            // Decide-and-mark must happen under one short-lived lock so two
            // concurrent loads on the same conversation_id can't both
            // observe the same `absorbed` watermark and run the compactor
            // twice with the same input slice.
            //
            // Fast path: if the conversation is already tracked, mutate in
            // place. Only allocate a new `String` key when there is real
            // compaction work for a conversation we have not seen before.
            let plan = {
                let mut guard = self.state.lock().map_err(poisoned)?;
                if let Some(entry) = guard.get_mut(conversation_id) {
                    if entry.in_flight.is_some() {
                        // Another load is mid-compaction; return what we
                        // have so far. Newly-evicted messages will be
                        // folded in by the next load.
                        return Ok(splice(entry.summary.clone(), kept));
                    }
                    if demoted_count <= entry.absorbed {
                        // No new evictions to compact. Splice the existing
                        // summary (if any) and we're done.
                        return Ok(splice(entry.summary.clone(), kept));
                    }
                    let reservation = Arc::new(());
                    entry.in_flight = Some(reservation.clone());
                    CompactionPlan {
                        carry_over: entry.summary.clone(),
                        skip: entry.absorbed,
                        reservation,
                    }
                } else if demoted_count == 0 {
                    // First load for this conversation and nothing was
                    // demoted: no tracking entry needed yet.
                    return Ok(kept);
                } else {
                    let reservation = Arc::new(());
                    guard.insert(
                        conversation_id.clone(),
                        ConversationCompactionState {
                            summary: None,
                            absorbed: 0,
                            in_flight: Some(reservation.clone()),
                        },
                    );
                    CompactionPlan {
                        carry_over: None,
                        skip: 0,
                        reservation,
                    }
                }
            };

            // SAFETY: split_at(plan.skip) is sound because `plan.skip` was
            // sourced from the entry's `absorbed` watermark while we held
            // the lock, and we only set `absorbed = demoted_count` on
            // success — so `plan.skip <= demoted_count == demoted.len()`.
            let CompactionPlan {
                carry_over,
                skip,
                reservation,
            } = plan;

            // Arm an RAII guard so the in-flight gate is released even if
            // this future is dropped mid-await (caller cancellation) or
            // the compactor panics. The guard is disarmed below once the
            // post-await state update has already cleared the flag under
            // the same lock acquisition that records the new watermark.
            let in_flight_guard =
                InFlightGuard::new(&self.state, conversation_id, reservation.clone());

            let Some(new_slice) = demoted.get(skip..) else {
                // Drop the guard explicitly so the gate is released
                // before we surface the invariant break.
                drop(in_flight_guard);
                return Err(MemoryError::Internal(
                    "compaction watermark exceeds demoted slice length".into(),
                ));
            };

            let result = self
                .compactor
                .compact(conversation_id, new_slice, carry_over.as_ref())
                .await;

            // Reacquire briefly to advance the watermark on success and
            // always clear the in-flight flag so a future load can retry.
            //
            // Only update if the entry still exists: a concurrent `clear`
            // (and matching `forget`) for this `conversation_id` may have
            // dropped the state entry while the compactor was awaiting. In
            // that case we must not resurrect it with stale state — the
            // next load on a freshly-populated backend would then start
            // from a non-zero watermark and skip a real compaction.
            // A conversation cleared mid-compaction has no entry anymore; the
            // artifact is dropped rather than reviving stale state.
            let summary_for_splice = match result {
                Ok(artifact) => {
                    release_in_flight(&self.state, conversation_id, &reservation, |entry| {
                        entry.absorbed = demoted_count;
                        entry.summary = Some(artifact.clone());
                        artifact
                    })?
                }
                Err(err) => {
                    release_in_flight(&self.state, conversation_id, &reservation, |_| ())?;
                    return Err(err);
                }
            };

            // Post-await state update completed under the lock above and
            // already cleared `in_flight`; disarm the RAII guard so its
            // `Drop` does not re-acquire the lock for a redundant clear.
            in_flight_guard.disarm();

            Ok(splice(summary_for_splice, kept))
        })
    }

    stateful_wrapper_append_clear!();
}

struct CompactionPlan<A> {
    carry_over: Option<A>,
    skip: usize,
    reservation: InFlightReservation,
}

fn splice<A>(summary: Option<A>, kept: Vec<Message>) -> Vec<Message>
where
    A: Into<Message>,
{
    match summary {
        Some(artifact) => {
            let mut out = Vec::with_capacity(kept.len() + 1);
            out.push(artifact.into());
            out.extend(kept);
            out
        }
        None => kept,
    }
}

/// A zero-dependency reference [`Compactor`] that produces a textual
/// rollup of evicted messages without calling an LLM.
///
/// The artifact is a single [`Message::System`] whose body concatenates a
/// header, the previous summary (if any), and the textual content of each
/// newly-evicted message. It is intentionally simple: useful as a default
/// for tests and examples, and as a placeholder before wiring a real
/// summarising LLM through a custom [`Compactor`] implementation.
///
/// # Bounding the summary
///
/// By default the summary grows monotonically: every compaction pass
/// embeds the previous summary verbatim and appends newly-evicted lines.
/// Long-running conversations should call [`Self::with_max_bytes`] to
/// cap the rolled-up text. When the cap is exceeded, the oldest portion
/// of the body (after the header) is dropped at a UTF-8 boundary and
/// replaced with a `"[…truncated…]"` marker, preserving the most recent
/// context.
///
/// # Example
///
/// ```
/// use rig_memory::TemplateCompactor;
///
/// // Default header is "[Conversation summary so far]", unbounded.
/// let _compactor = TemplateCompactor::new();
///
/// // Custom header plus a 4 KiB cap for use with token-budgeted policies.
/// let _bounded = TemplateCompactor::with_header("Earlier context")
///     .with_max_bytes(4 * 1024);
/// ```
#[derive(Debug, Clone)]
pub struct TemplateCompactor {
    header: String,
    max_bytes: Option<usize>,
}

impl TemplateCompactor {
    /// Create a [`TemplateCompactor`] with the default header
    /// `"[Conversation summary so far]"` and no size cap.
    pub fn new() -> Self {
        Self::with_header("[Conversation summary so far]")
    }

    /// Create a [`TemplateCompactor`] with a custom header line and no
    /// size cap.
    pub fn with_header(header: impl Into<String>) -> Self {
        Self {
            header: header.into(),
            max_bytes: None,
        }
    }

    /// Cap the rolled-up summary at `max_bytes` bytes (UTF-8). When the
    /// assembled body exceeds the cap, the oldest portion after the
    /// header is dropped at a char boundary and replaced with a
    /// `"[…truncated…]"` marker.
    ///
    /// `max_bytes` of `0` disables truncation (equivalent to the default
    /// unbounded behaviour). The header line plus the marker are always
    /// preserved even if they exceed the cap.
    pub fn with_max_bytes(mut self, max_bytes: usize) -> Self {
        self.max_bytes = if max_bytes == 0 {
            None
        } else {
            Some(max_bytes)
        };
        self
    }
}

impl Default for TemplateCompactor {
    fn default() -> Self {
        Self::new()
    }
}

/// Plain-text artifact produced by [`TemplateCompactor`].
///
/// Convertible into a [`Message::System`] whose body is the rolled-up
/// text. The system role is used because the rollup represents
/// out-of-band context about the prior conversation, not a turn from
/// any participant.
#[derive(Debug, Clone)]
pub struct TextSummary(String);

impl TextSummary {
    /// Borrow the underlying summary text.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume the wrapper and return the underlying `String`.
    pub fn into_string(self) -> String {
        self.0
    }
}

impl From<TextSummary> for Message {
    fn from(value: TextSummary) -> Self {
        Message::System { content: value.0 }
    }
}

impl Compactor for TemplateCompactor {
    type Artifact = TextSummary;

    fn compact<'a>(
        &'a self,
        _conversation_id: &'a ConversationId,
        evicted: &'a [Message],
        carry_over: Option<&'a Self::Artifact>,
    ) -> WasmBoxedFuture<'a, Result<Self::Artifact, MemoryError>> {
        Box::pin(async move {
            let mut buf = String::new();
            buf.push_str(&self.header);
            buf.push('\n');
            if let Some(prev) = carry_over {
                buf.push_str(prev.as_str());
                buf.push('\n');
            }
            for msg in evicted {
                let line = render_message_line(msg);
                if !line.is_empty() {
                    buf.push_str(&line);
                    buf.push('\n');
                }
            }
            if let Some(cap) = self.max_bytes
                && buf.len() > cap
            {
                buf = truncate_summary(&buf, cap);
            }
            Ok(TextSummary(buf))
        })
    }
}

/// Truncate `buf` to fit within `cap` bytes by dropping the oldest
/// content after the header line. Always preserves the header plus a
/// `"[\u{2026}truncated\u{2026}]"` marker, even if they alone exceed `cap`.
///
/// The header boundary is located by scanning `buf` for the first `\n`
/// rather than by trusting any caller-supplied header length, so a
/// header containing embedded newlines does not mis-locate the body.
fn truncate_summary(buf: &str, cap: usize) -> String {
    const MARKER: &str = "[\u{2026}truncated\u{2026}]\n";
    // Body starts right after the first newline in `buf`. If `buf` has
    // no newline at all there is no body to drop, so return as-is.
    let Some(newline) = buf.find('\n') else {
        return buf.to_string();
    };
    let header_prefix_len = newline + 1;
    if buf.len() <= header_prefix_len {
        return buf.to_string();
    }
    let preserved = header_prefix_len + MARKER.len();
    // Number of bytes of the body we can keep after the marker.
    let keep_bytes = cap.saturating_sub(preserved);
    let body_start = header_prefix_len;
    let Some(body) = buf.get(body_start..) else {
        return buf.to_string();
    };
    // Take the suffix of `body` whose length is at most `keep_bytes`,
    // walking forward to a UTF-8 char boundary.
    let mut cut = body.len().saturating_sub(keep_bytes);
    while cut < body.len() && !body.is_char_boundary(cut) {
        cut += 1;
    }
    let suffix: &str = body.get(cut..).unwrap_or_default();
    let Some(header_with_nl) = buf.get(..header_prefix_len) else {
        return buf.to_string();
    };
    let mut out = String::with_capacity(header_prefix_len + MARKER.len() + suffix.len());
    out.push_str(header_with_nl);
    out.push_str(MARKER);
    out.push_str(suffix);
    out
}

/// Render a single message as a `"role: text"` line for [`TemplateCompactor`].
///
/// Non-textual content (tool calls, tool results, attachments) is rendered
/// as a short marker so the rollup does not silently drop them but also
/// does not balloon with serialized JSON.
fn render_message_line(msg: &Message) -> String {
    use rig_core::message::AssistantContent;

    let (role, text) = match msg {
        Message::System { content } => ("system", content.clone()),
        Message::User { content } => (
            "user",
            join_parts(content.iter().map(|item| match item {
                UserContent::Text(text) => Cow::Borrowed(text.text.as_str()),
                UserContent::ToolResult(_) => Cow::Borrowed("[tool result]"),
                _ => Cow::Borrowed("[attachment]"),
            })),
        ),
        Message::Assistant { content, .. } => (
            "assistant",
            join_parts(content.iter().map(|item| match item {
                AssistantContent::Text(text) => Cow::Borrowed(text.text.as_str()),
                AssistantContent::ToolCall(call) => {
                    Cow::Owned(format!("[tool call: {}]", call.function.name))
                }
                _ => Cow::Borrowed("[reasoning]"),
            })),
        ),
    };

    if text.is_empty() {
        String::new()
    } else {
        format!("{role}: {text}")
    }
}

/// Space-join rendered parts, suppressing the separator while the line is
/// still empty.
///
/// Not `Vec::join(" ")`: that would emit a leading separator for a message
/// whose first part renders empty, while still collapsing nothing elsewhere.
/// The rollup text these lines feed is compared byte-for-byte by the
/// compaction tests, so the asymmetry is behavior, not an accident.
fn join_parts<'a>(parts: impl Iterator<Item = Cow<'a, str>>) -> String {
    let mut text = String::new();
    for part in parts {
        if !text.is_empty() {
            text.push(' ');
        }
        text.push_str(&part);
    }
    text
}

#[cfg(test)]
mod tests;
