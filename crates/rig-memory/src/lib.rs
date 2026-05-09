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
    #[cfg(not(target_family = "wasm"))]
    fn into_filter(self) -> Box<dyn Fn(Vec<Message>) -> Vec<Message> + Send + Sync> {
        let policy = Arc::new(self);
        Box::new(move |msgs| {
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

    /// Convert this policy into a filter closure.
    ///
    /// On policy error the original input is returned unchanged and a
    /// `tracing::warn!` is emitted, so a transient policy bug degrades
    /// gracefully (the model still sees the unfiltered history) instead of
    /// silently erasing context.
    #[cfg(target_family = "wasm")]
    fn into_filter(self) -> Box<dyn Fn(Vec<Message>) -> Vec<Message>> {
        let policy = Arc::new(self);
        Box::new(move |msgs| {
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

        let start = messages.len() - self.max_messages;
        let mut iter = messages.into_iter();
        let mut demoted: Vec<Message> = (&mut iter).take(start).collect();
        let mut window: Vec<Message> = iter.collect();

        // The orphan tool-result, if any, becomes part of the demoted set so
        // it is preserved end-to-end through the demotion hook even though
        // the model never sees it again.
        if let Some(Message::User { content }) = window.first()
            && matches!(content.first_ref(), UserContent::ToolResult(_))
        {
            demoted.push(window.remove(0));
        }

        Ok((window, demoted))
    }
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

        let mut iter = messages.into_iter();
        let mut demoted: Vec<Message> = (&mut iter).take(keep_from).collect();
        let mut window: Vec<Message> = iter.collect();

        if let Some(Message::User { content }) = window.first()
            && matches!(content.first_ref(), UserContent::ToolResult(_))
        {
            demoted.push(window.remove(0));
        }

        Ok((window, demoted))
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
        conversation_id: &'a str,
    ) -> WasmBoxedFuture<'a, Result<Vec<Message>, MemoryError>> {
        Box::pin(async move {
            let messages = self.inner.load(conversation_id).await?;
            self.policy.apply(messages)
        })
    }

    fn append<'a>(
        &'a self,
        conversation_id: &'a str,
        messages: Vec<Message>,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
        self.inner.append(conversation_id, messages)
    }

    fn clear<'a>(
        &'a self,
        conversation_id: &'a str,
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
    state: StdMutex<HashMap<String, ConversationDemotionState>>,
}

#[derive(Debug, Default, Clone, Copy)]
struct ConversationDemotionState {
    /// Number of demoted messages already delivered to the hook within
    /// this process lifetime. Advanced only on hook success.
    delivered: usize,
    /// True while a `load` is currently awaiting `hook.on_demote(...)`
    /// for this conversation. Other concurrent loads observe this and
    /// short-circuit without re-delivering the same messages.
    in_flight: bool,
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

    /// Return a reference to the wrapped backend.
    pub fn inner(&self) -> &M {
        &self.inner
    }

    /// Return a reference to the wrapped policy.
    pub fn policy(&self) -> &P {
        &self.policy
    }

    /// Return a reference to the demotion hook.
    pub fn hook(&self) -> &H {
        &self.hook
    }

    /// Consume the wrapper and return its three components.
    pub fn into_inner(self) -> (M, P, H) {
        (self.inner, self.policy, self.hook)
    }

    /// Drop the in-process delivery watermark for `conversation_id`.
    ///
    /// Call this when a conversation has ended to bound memory usage.
    /// The watermark map is otherwise unbounded — entries persist for
    /// the lifetime of the wrapper.
    ///
    /// If the internal state lock has been poisoned by a panic in another
    /// thread, this is a no-op (the watermark will be dropped naturally
    /// when the wrapper itself is dropped).
    pub fn forget(&self, conversation_id: &str) {
        if let Ok(mut guard) = self.state.lock() {
            guard.remove(conversation_id);
        }
    }

    /// Number of conversations currently tracked in the watermark map.
    /// Useful for telemetry and leak detection. Returns `0` if the internal
    /// state lock is poisoned.
    pub fn tracked_conversations(&self) -> usize {
        self.state.lock().map(|g| g.len()).unwrap_or(0)
    }
}

impl<M, P, H> std::fmt::Debug for DemotingPolicyMemory<M, P, H>
where
    M: std::fmt::Debug,
    P: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DemotingPolicyMemory")
            .field("inner", &self.inner)
            .field("policy", &self.policy)
            .field("hook", &"<hook>")
            .finish()
    }
}

impl<M, P, H> ConversationMemory for DemotingPolicyMemory<M, P, H>
where
    M: ConversationMemory,
    P: MemoryPolicy,
    H: DemotionHook,
{
    fn load<'a>(
        &'a self,
        conversation_id: &'a str,
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
            let pending = {
                let mut guard = self.state.lock().map_err(poisoned)?;
                if let Some(entry) = guard.get_mut(conversation_id) {
                    if entry.in_flight {
                        // Another load is mid-delivery for this conversation;
                        // skip and let the next load see whatever it leaves
                        // behind.
                        return Ok(kept);
                    }
                    if entry.delivered >= demoted_count {
                        Vec::new()
                    } else {
                        let split = entry.delivered;
                        entry.in_flight = true;
                        demoted.split_off(split)
                    }
                } else if demoted_count == 0 {
                    // First load for this conversation and nothing was
                    // demoted: no need to allocate a tracking entry yet.
                    Vec::new()
                } else {
                    guard.insert(
                        conversation_id.to_string(),
                        ConversationDemotionState {
                            delivered: 0,
                            in_flight: true,
                        },
                    );
                    std::mem::take(&mut demoted)
                }
            };

            if pending.is_empty() {
                return Ok(kept);
            }

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
            {
                let mut guard = self.state.lock().map_err(poisoned)?;
                if let Some(entry) = guard.get_mut(conversation_id) {
                    entry.in_flight = false;
                    if result.is_ok() {
                        entry.delivered = demoted_count;
                    }
                }
            }
            result?;
            Ok(kept)
        })
    }

    fn append<'a>(
        &'a self,
        conversation_id: &'a str,
        messages: Vec<Message>,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
        self.inner.append(conversation_id, messages)
    }

    fn clear<'a>(
        &'a self,
        conversation_id: &'a str,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
        Box::pin(async move {
            self.inner.clear(conversation_id).await?;
            self.forget(conversation_id);
            Ok(())
        })
    }
}

fn poisoned<E: std::fmt::Display>(err: E) -> MemoryError {
    MemoryError::Internal(err.to_string())
}

/// RAII guard that clears the `in_flight` flag for a conversation in the
/// shared compaction state map when dropped, unless the consumer
/// explicitly disarms it after a successful post-await update.
///
/// This prevents the in-flight gate from leaking when the awaiting
/// `load(...)` future is dropped (caller timeout, `tokio::select!`, etc.)
/// or when the compactor panics: in either case `Drop` runs and releases
/// the gate so subsequent loads can retry. A missing entry is a no-op,
/// covering the case where a concurrent `clear` removed the conversation
/// while compaction was awaiting.
struct InFlightGuard<'a, A> {
    state: &'a StdMutex<HashMap<String, ConversationCompactionState<A>>>,
    key: &'a str,
    armed: bool,
}

impl<'a, A> InFlightGuard<'a, A> {
    fn new(
        state: &'a StdMutex<HashMap<String, ConversationCompactionState<A>>>,
        key: &'a str,
    ) -> Self {
        Self {
            state,
            key,
            armed: true,
        }
    }

    /// Disable the `Drop` clean-up. Call after the post-await state
    /// update has already cleared `in_flight` while holding the lock.
    fn disarm(mut self) {
        self.armed = false;
    }
}

impl<A> Drop for InFlightGuard<'_, A> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        if let Ok(mut guard) = self.state.lock()
            && let Some(entry) = guard.get_mut(self.key)
        {
            entry.in_flight = false;
        }
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
    state: StdMutex<HashMap<String, ConversationCompactionState<C::Artifact>>>,
}

struct ConversationCompactionState<A> {
    /// Latest summary artifact for this conversation, if compaction has
    /// already happened. Cloned into the loaded history on every `load`.
    summary: Option<A>,
    /// Number of demoted messages already absorbed into `summary` within
    /// this process lifetime. Advanced only on compactor success.
    absorbed: usize,
    /// True while a `load` is currently awaiting the compactor for this
    /// conversation. Other concurrent loads observe this and short-circuit
    /// without re-running the compactor.
    in_flight: bool,
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

    /// Return a reference to the wrapped backend.
    pub fn inner(&self) -> &M {
        &self.inner
    }

    /// Return a reference to the wrapped policy.
    pub fn policy(&self) -> &P {
        &self.policy
    }

    /// Return a reference to the compactor.
    pub fn compactor(&self) -> &C {
        &self.compactor
    }

    /// Consume the wrapper and return its three components.
    pub fn into_inner(self) -> (M, P, C) {
        (self.inner, self.policy, self.compactor)
    }

    /// Drop the in-process compaction state for `conversation_id`.
    ///
    /// Call this when a conversation has ended to bound memory usage; the
    /// state map is otherwise unbounded. If the internal lock has been
    /// poisoned by a panic in another thread, this is a no-op.
    pub fn forget(&self, conversation_id: &str) {
        if let Ok(mut guard) = self.state.lock() {
            guard.remove(conversation_id);
        }
    }

    /// Number of conversations currently tracked in the compaction state
    /// map. Useful for telemetry and leak detection. Returns `0` if the
    /// internal lock is poisoned.
    pub fn tracked_conversations(&self) -> usize {
        self.state.lock().map(|g| g.len()).unwrap_or(0)
    }
}

impl<M, P, C> std::fmt::Debug for CompactingMemory<M, P, C>
where
    M: std::fmt::Debug,
    P: std::fmt::Debug,
    C: Compactor,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompactingMemory")
            .field("inner", &self.inner)
            .field("policy", &self.policy)
            .field("compactor", &"<compactor>")
            .finish()
    }
}

impl<M, P, C> ConversationMemory for CompactingMemory<M, P, C>
where
    M: ConversationMemory,
    P: MemoryPolicy,
    C: Compactor,
{
    fn load<'a>(
        &'a self,
        conversation_id: &'a str,
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
                    if entry.in_flight {
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
                    entry.in_flight = true;
                    CompactionPlan {
                        carry_over: entry.summary.clone(),
                        skip: entry.absorbed,
                    }
                } else if demoted_count == 0 {
                    // First load for this conversation and nothing was
                    // demoted: no tracking entry needed yet.
                    return Ok(kept);
                } else {
                    guard.insert(
                        conversation_id.to_string(),
                        ConversationCompactionState {
                            summary: None,
                            absorbed: 0,
                            in_flight: true,
                        },
                    );
                    CompactionPlan {
                        carry_over: None,
                        skip: 0,
                    }
                }
            };

            // SAFETY: split_at(plan.skip) is sound because `plan.skip` was
            // sourced from the entry's `absorbed` watermark while we held
            // the lock, and we only set `absorbed = demoted_count` on
            // success — so `plan.skip <= demoted_count == demoted.len()`.
            let CompactionPlan { carry_over, skip } = plan;

            // Arm an RAII guard so the in-flight gate is released even if
            // this future is dropped mid-await (caller cancellation) or
            // the compactor panics. The guard is disarmed below once the
            // post-await state update has already cleared the flag under
            // the same lock acquisition that records the new watermark.
            let in_flight_guard = InFlightGuard::new(&self.state, conversation_id);

            let new_slice = match demoted.get(skip..) {
                Some(s) => s,
                None => {
                    // Drop the guard explicitly so the gate is released
                    // before we surface the invariant break.
                    drop(in_flight_guard);
                    return Err(MemoryError::Internal(
                        "compaction watermark exceeds demoted slice length".into(),
                    ));
                }
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
            let summary_for_splice = match result {
                Ok(artifact) => {
                    let mut guard = self.state.lock().map_err(poisoned)?;
                    if let Some(entry) = guard.get_mut(conversation_id) {
                        entry.in_flight = false;
                        entry.absorbed = demoted_count;
                        entry.summary = Some(artifact.clone());
                        Some(artifact)
                    } else {
                        // Conversation was cleared mid-compaction. Drop
                        // the artifact rather than reviving stale state.
                        None
                    }
                }
                Err(err) => {
                    let mut guard = self.state.lock().map_err(poisoned)?;
                    if let Some(entry) = guard.get_mut(conversation_id) {
                        entry.in_flight = false;
                    }
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

    fn append<'a>(
        &'a self,
        conversation_id: &'a str,
        messages: Vec<Message>,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
        self.inner.append(conversation_id, messages)
    }

    fn clear<'a>(
        &'a self,
        conversation_id: &'a str,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
        Box::pin(async move {
            self.inner.clear(conversation_id).await?;
            self.forget(conversation_id);
            Ok(())
        })
    }
}

struct CompactionPlan<A> {
    carry_over: Option<A>,
    skip: usize,
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
        _conversation_id: &'a str,
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
    let header_prefix_len = match buf.find('\n') {
        Some(i) => i + 1,
        None => return buf.to_string(),
    };
    if buf.len() <= header_prefix_len {
        return buf.to_string();
    }
    let preserved = header_prefix_len + MARKER.len();
    // Number of bytes of the body we can keep after the marker.
    let keep_bytes = cap.saturating_sub(preserved);
    let body_start = header_prefix_len;
    let body = match buf.get(body_start..) {
        Some(b) => b,
        None => return buf.to_string(),
    };
    // Take the suffix of `body` whose length is at most `keep_bytes`,
    // walking forward to a UTF-8 char boundary.
    let mut cut = body.len().saturating_sub(keep_bytes);
    while cut < body.len() && !body.is_char_boundary(cut) {
        cut += 1;
    }
    let suffix = match body.get(cut..) {
        Some(s) => s,
        None => "",
    };
    let header_with_nl = match buf.get(..header_prefix_len) {
        Some(h) => h,
        None => return buf.to_string(),
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

    match msg {
        Message::System { content } => {
            if content.is_empty() {
                String::new()
            } else {
                format!("system: {content}")
            }
        }
        Message::User { content } => {
            let mut text = String::new();
            for c in content.iter() {
                match c {
                    UserContent::Text(t) => {
                        if !text.is_empty() {
                            text.push(' ');
                        }
                        text.push_str(&t.text);
                    }
                    UserContent::ToolResult(_) => {
                        if !text.is_empty() {
                            text.push(' ');
                        }
                        text.push_str("[tool result]");
                    }
                    _ => {
                        if !text.is_empty() {
                            text.push(' ');
                        }
                        text.push_str("[attachment]");
                    }
                }
            }
            if text.is_empty() {
                String::new()
            } else {
                format!("user: {text}")
            }
        }
        Message::Assistant { content, .. } => {
            let mut text = String::new();
            for c in content.iter() {
                match c {
                    AssistantContent::Text(t) => {
                        if !text.is_empty() {
                            text.push(' ');
                        }
                        text.push_str(&t.text);
                    }
                    AssistantContent::ToolCall(call) => {
                        if !text.is_empty() {
                            text.push(' ');
                        }
                        text.push_str(&format!("[tool call: {}]", call.function.name));
                    }
                    _ => {
                        if !text.is_empty() {
                            text.push(' ');
                        }
                        text.push_str("[reasoning]");
                    }
                }
            }
            if text.is_empty() {
                String::new()
            } else {
                format!("assistant: {text}")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rig_core::OneOrMany;
    use rig_core::message::{
        AssistantContent, ToolCall, ToolFunction, ToolResult, ToolResultContent, UserContent,
    };
    use std::sync::Mutex;

    fn user(text: &str) -> Message {
        Message::user(text)
    }

    fn assistant(text: &str) -> Message {
        Message::assistant(text)
    }

    fn tool_call_msg() -> Message {
        Message::Assistant {
            id: None,
            content: OneOrMany::one(AssistantContent::ToolCall(ToolCall::new(
                "call_1".into(),
                ToolFunction::new("t".into(), serde_json::json!({})),
            ))),
        }
    }

    fn tool_result_msg() -> Message {
        Message::User {
            content: OneOrMany::one(UserContent::ToolResult(ToolResult {
                id: "call_1".into(),
                call_id: None,
                content: OneOrMany::one(ToolResultContent::text("ok")),
            })),
        }
    }

    #[test]
    fn noop_policy_is_identity() {
        let msgs = vec![user("a"), assistant("b")];
        let out = NoopMemoryPolicy.apply(msgs).unwrap();
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn sliding_window_passthrough_when_under_limit() {
        let policy = SlidingWindowMemory::last_messages(5);
        let out = policy.apply(vec![user("1"), assistant("2")]).unwrap();
        assert_eq!(out.len(), 2);
    }

    #[tokio::test]
    async fn sliding_window_truncates_via_filter() {
        let mem = InMemoryConversationMemory::new()
            .with_filter(SlidingWindowMemory::last_messages(2).into_filter());

        mem.append(
            "c",
            vec![user("1"), assistant("2"), user("3"), assistant("4")],
        )
        .await
        .unwrap();

        let loaded = mem.load("c").await.unwrap();
        assert_eq!(loaded.len(), 2);
    }

    #[test]
    fn sliding_window_drops_leading_orphan_tool_result() {
        let policy = SlidingWindowMemory::last_messages(3);
        let out = policy
            .apply(vec![
                tool_call_msg(),
                tool_result_msg(),
                user("after"),
                assistant("done"),
            ])
            .unwrap();

        assert_eq!(out.len(), 2);
        assert!(matches!(out.first(), Some(Message::User { content })
            if matches!(content.first(), UserContent::Text(_))));
    }

    #[test]
    fn token_window_keeps_within_budget() {
        let msgs = vec![
            user("aaaa"),
            assistant("bbbb"),
            user("cccc"),
            assistant("dddd"),
        ];
        let policy = TokenWindowMemory::new(2, |_: &Message| 1);
        let out = policy.apply(msgs).unwrap();
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn token_window_passes_through_when_under_budget() {
        let msgs = vec![user("a"), assistant("b")];
        let policy = TokenWindowMemory::new(usize::MAX, |_: &Message| 1);
        let out = policy.apply(msgs).unwrap();
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn token_window_drops_leading_orphan_tool_result() {
        let policy = TokenWindowMemory::new(25, |_: &Message| 10);
        let out = policy
            .apply(vec![tool_call_msg(), tool_result_msg(), user("after")])
            .unwrap();
        assert_eq!(out.len(), 1);
        assert!(matches!(out.first(), Some(Message::User { content })
            if matches!(content.first(), UserContent::Text(_))));
    }

    #[test]
    fn token_window_skips_message_larger_than_budget() {
        let policy = TokenWindowMemory::new(5, |_: &Message| 10);
        let out = policy.apply(vec![user("anything")]).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn heuristic_counter_charges_overhead_per_message() {
        let counter = HeuristicTokenCounter::default();
        let empty = counter.count(&user(""));
        assert!(
            empty >= 4,
            "default per-message overhead is at least 4 tokens"
        );
    }

    #[test]
    fn heuristic_counter_is_monotonic_in_text_length() {
        let counter = HeuristicTokenCounter::default();
        let small = counter.count(&user("hi"));
        let big = counter.count(&user(&"x".repeat(400)));
        assert!(big > small);
    }

    #[test]
    fn heuristic_counter_handles_tool_calls() {
        let counter = HeuristicTokenCounter::default();
        let cost = counter.count(&tool_call_msg());
        assert!(cost > 0);
    }

    #[test]
    fn heuristic_counter_handles_system_messages() {
        let counter = HeuristicTokenCounter::default();
        let cost = counter.count(&Message::System {
            content: "you are helpful".into(),
        });
        assert!(cost > 0);
    }

    #[test]
    fn heuristic_counter_clamps_invalid_bytes_per_token() {
        // Zero/NaN/negative ratios fall back to 1.0 instead of panicking.
        let counter = HeuristicTokenCounter::new(0.0, 0, 0);
        assert!(counter.count(&user("abcd")) >= 4);
        let nan = HeuristicTokenCounter::new(f32::NAN, 0, 0);
        assert!(nan.count(&user("abcd")) >= 4);
    }

    #[test]
    fn heuristic_counter_drives_token_window() {
        let policy = TokenWindowMemory::new(100, HeuristicTokenCounter::default());
        let msgs = vec![user(&"a".repeat(2_000)), user("short")];
        let out = policy.apply(msgs).unwrap();
        // The huge message must be evicted; the short one retained.
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn into_filter_returns_input_on_policy_error() {
        struct FailingPolicy;
        impl MemoryPolicy for FailingPolicy {
            fn apply(&self, _: Vec<Message>) -> Result<Vec<Message>, MemoryError> {
                Err(MemoryError::Policy("intentional failure".into()))
            }
        }

        let filter = FailingPolicy.into_filter();
        let input = vec![user("a"), assistant("b"), user("c")];
        let out = filter(input.clone());
        assert_eq!(
            out.len(),
            input.len(),
            "history must be preserved on policy error"
        );
    }

    #[tokio::test]
    async fn policy_memory_truncates_loaded_history() {
        let mem = PolicyMemory::new(
            InMemoryConversationMemory::new(),
            SlidingWindowMemory::last_messages(2),
        );

        mem.append(
            "c",
            vec![user("1"), assistant("2"), user("3"), assistant("4")],
        )
        .await
        .unwrap();

        let loaded = mem.load("c").await.unwrap();
        assert_eq!(loaded.len(), 2);
    }

    #[tokio::test]
    async fn policy_memory_propagates_policy_errors() {
        struct FailingPolicy;
        impl MemoryPolicy for FailingPolicy {
            fn apply(&self, _: Vec<Message>) -> Result<Vec<Message>, MemoryError> {
                Err(MemoryError::Policy("intentional failure".into()))
            }
        }

        let mem = PolicyMemory::new(InMemoryConversationMemory::new(), FailingPolicy);
        mem.append("c", vec![user("1"), assistant("2")])
            .await
            .unwrap();

        let result = mem.load("c").await;
        assert!(matches!(result, Err(MemoryError::Policy(_))));
    }

    #[tokio::test]
    async fn policy_memory_append_and_clear_delegate_to_inner() {
        let mem = PolicyMemory::new(InMemoryConversationMemory::new(), NoopMemoryPolicy);
        mem.append("c", vec![user("hi"), assistant("ok")])
            .await
            .unwrap();
        assert_eq!(mem.load("c").await.unwrap().len(), 2);

        mem.clear("c").await.unwrap();
        assert!(mem.load("c").await.unwrap().is_empty());
    }

    #[test]
    fn sliding_window_reports_demoted_prefix() {
        let policy = SlidingWindowMemory::last_messages(2);
        let (kept, demoted) = policy
            .apply_with_demoted(vec![
                user("oldest"),
                assistant("old"),
                user("recent"),
                assistant("latest"),
            ])
            .unwrap();
        assert_eq!(kept.len(), 2);
        assert_eq!(demoted.len(), 2);
    }

    #[test]
    fn token_window_reports_demoted_prefix() {
        let policy = TokenWindowMemory::new(2, |_: &Message| 1);
        let (kept, demoted) = policy
            .apply_with_demoted(vec![user("a"), assistant("b"), user("c"), assistant("d")])
            .unwrap();
        assert_eq!(kept.len(), 2);
        assert_eq!(demoted.len(), 2);
    }

    #[test]
    fn noop_policy_demotes_nothing() {
        let (kept, demoted) = NoopMemoryPolicy
            .apply_with_demoted(vec![user("a"), assistant("b")])
            .unwrap();
        assert_eq!(kept.len(), 2);
        assert!(demoted.is_empty());
    }

    #[test]
    fn sliding_window_demotes_orphan_tool_result_with_prefix() {
        // Window keeps the last 2 messages, but the leading message of that
        // window is an orphan tool result; it must be moved into `demoted`
        // so the hook can preserve it.
        let policy = SlidingWindowMemory::last_messages(2);
        let (kept, demoted) = policy
            .apply_with_demoted(vec![
                tool_call_msg(),
                tool_result_msg(),
                user("after"),
                assistant("done"),
            ])
            .unwrap();
        assert_eq!(kept.len(), 2);
        assert!(matches!(kept.first(), Some(Message::User { content })
            if matches!(content.first(), UserContent::Text(_))));
        assert_eq!(demoted.len(), 2);
    }

    #[derive(Default)]
    struct CountingHook {
        seen: Mutex<Vec<(String, Vec<Message>)>>,
    }

    impl CountingHook {
        fn calls(&self) -> usize {
            self.seen.lock().unwrap().len()
        }
        fn last_demoted_count(&self) -> usize {
            self.seen
                .lock()
                .unwrap()
                .last()
                .map(|(_, m)| m.len())
                .unwrap_or(0)
        }
    }

    impl DemotionHook for CountingHook {
        fn on_demote<'a>(
            &'a self,
            conversation_id: &'a str,
            messages: Vec<Message>,
        ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
            Box::pin(async move {
                self.seen
                    .lock()
                    .unwrap()
                    .push((conversation_id.to_string(), messages));
                Ok(())
            })
        }
    }

    #[tokio::test]
    async fn demoting_policy_memory_invokes_hook_on_truncation() {
        let hook = Arc::new(CountingHook::default());
        let mem = DemotingPolicyMemory::new(
            InMemoryConversationMemory::new(),
            SlidingWindowMemory::last_messages(2),
            hook.clone(),
        );

        mem.append(
            "c",
            vec![user("1"), assistant("2"), user("3"), assistant("4")],
        )
        .await
        .unwrap();

        let kept = mem.load("c").await.unwrap();
        assert_eq!(kept.len(), 2);
        assert_eq!(hook.calls(), 1);
        assert_eq!(hook.last_demoted_count(), 2);
    }

    #[tokio::test]
    async fn demoting_policy_memory_does_not_replay_demotions() {
        let hook = Arc::new(CountingHook::default());
        let mem = DemotingPolicyMemory::new(
            InMemoryConversationMemory::new(),
            SlidingWindowMemory::last_messages(2),
            hook.clone(),
        );

        mem.append(
            "c",
            vec![user("1"), assistant("2"), user("3"), assistant("4")],
        )
        .await
        .unwrap();

        mem.load("c").await.unwrap();
        mem.load("c").await.unwrap();
        assert_eq!(hook.calls(), 1);
        assert_eq!(hook.last_demoted_count(), 2);
    }

    #[tokio::test]
    async fn demoting_policy_memory_only_reports_newly_demoted_messages() {
        let hook = Arc::new(CountingHook::default());
        let mem = DemotingPolicyMemory::new(
            InMemoryConversationMemory::new(),
            SlidingWindowMemory::last_messages(2),
            hook.clone(),
        );

        mem.append(
            "c",
            vec![user("1"), assistant("2"), user("3"), assistant("4")],
        )
        .await
        .unwrap();
        mem.load("c").await.unwrap();

        mem.append("c", vec![user("5")]).await.unwrap();
        mem.load("c").await.unwrap();

        assert_eq!(hook.calls(), 2);
        assert_eq!(hook.last_demoted_count(), 1);
    }

    #[derive(Default)]
    struct FailingHook {
        calls: Mutex<usize>,
    }

    impl DemotionHook for FailingHook {
        fn on_demote<'a>(
            &'a self,
            _conversation_id: &'a str,
            _messages: Vec<Message>,
        ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
            Box::pin(async move {
                *self.calls.lock().unwrap() += 1;
                Err(MemoryError::backend(std::io::Error::other("hook failed")))
            })
        }
    }

    #[tokio::test]
    async fn demoting_policy_memory_does_not_advance_watermark_on_hook_failure() {
        let hook = Arc::new(FailingHook::default());
        let mem = DemotingPolicyMemory::new(
            InMemoryConversationMemory::new(),
            SlidingWindowMemory::last_messages(1),
            hook.clone(),
        );
        mem.append("c", vec![user("1"), assistant("2")])
            .await
            .unwrap();

        assert!(mem.load("c").await.is_err());
        assert!(mem.load("c").await.is_err());
        assert_eq!(*hook.calls.lock().unwrap(), 2);
    }

    #[tokio::test]
    async fn demoting_policy_memory_clear_resets_watermark() {
        let hook = Arc::new(CountingHook::default());
        let mem = DemotingPolicyMemory::new(
            InMemoryConversationMemory::new(),
            SlidingWindowMemory::last_messages(1),
            hook.clone(),
        );

        mem.append("c", vec![user("1"), assistant("2")])
            .await
            .unwrap();
        mem.load("c").await.unwrap();
        mem.clear("c").await.unwrap();
        mem.append("c", vec![user("3"), assistant("4")])
            .await
            .unwrap();
        mem.load("c").await.unwrap();

        assert_eq!(hook.calls(), 2);
        assert_eq!(hook.last_demoted_count(), 1);
    }

    #[tokio::test]
    async fn demoting_policy_memory_skips_hook_when_nothing_evicted() {
        let hook = Arc::new(CountingHook::default());
        let mem = DemotingPolicyMemory::new(
            InMemoryConversationMemory::new(),
            SlidingWindowMemory::last_messages(10),
            hook.clone(),
        );

        mem.append("c", vec![user("1"), assistant("2")])
            .await
            .unwrap();
        mem.load("c").await.unwrap();
        assert_eq!(hook.calls(), 0);
    }

    #[tokio::test]
    async fn demoting_policy_memory_with_noop_hook_behaves_like_policy_memory() {
        let mem = DemotingPolicyMemory::new(
            InMemoryConversationMemory::new(),
            SlidingWindowMemory::last_messages(1),
            NoopDemotionHook,
        );
        mem.append("c", vec![user("a"), assistant("b"), user("c")])
            .await
            .unwrap();
        assert_eq!(mem.load("c").await.unwrap().len(), 1);
    }

    /// Hook that blocks until the test releases it. Used to provoke the
    /// concurrent-load race against the in-flight gate.
    struct GatedHook {
        calls: Arc<std::sync::atomic::AtomicUsize>,
        rendezvous: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
    }

    impl DemotionHook for GatedHook {
        fn on_demote<'a>(
            &'a self,
            _conversation_id: &'a str,
            _messages: Vec<Message>,
        ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
            let calls = self.calls.clone();
            let rendezvous = self.rendezvous.clone();
            let release = self.release.clone();
            Box::pin(async move {
                calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                rendezvous.notify_one();
                release.notified().await;
                Ok(())
            })
        }
    }

    #[tokio::test]
    async fn demoting_policy_memory_serialises_concurrent_loads() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let calls = Arc::new(AtomicUsize::new(0));
        let rendezvous = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let hook = GatedHook {
            calls: calls.clone(),
            rendezvous: rendezvous.clone(),
            release: release.clone(),
        };

        let mem = Arc::new(DemotingPolicyMemory::new(
            InMemoryConversationMemory::new(),
            SlidingWindowMemory::last_messages(1),
            hook,
        ));

        mem.append("c", vec![user("1"), assistant("2"), user("3")])
            .await
            .unwrap();

        let m1 = mem.clone();
        let first = tokio::spawn(async move { m1.load("c").await });

        // Wait until the first load has entered the hook.
        rendezvous.notified().await;
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        // Second concurrent load on the same conversation must skip the
        // hook entirely (in-flight gate) and return the truncated view.
        let kept = mem.load("c").await.unwrap();
        assert_eq!(kept.len(), 1);
        assert_eq!(calls.load(Ordering::SeqCst), 1, "hook must not double-fire");

        // Release the first load and confirm it completes successfully.
        release.notify_one();
        let kept_first = first.await.unwrap().unwrap();
        assert_eq!(kept_first.len(), 1);
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        // Subsequent loads observe the watermark and don't re-fire.
        mem.load("c").await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn forget_drops_in_process_watermark() {
        let hook = Arc::new(CountingHook::default());
        let mem = DemotingPolicyMemory::new(
            InMemoryConversationMemory::new(),
            SlidingWindowMemory::last_messages(1),
            hook.clone(),
        );

        mem.append("c", vec![user("1"), assistant("2")])
            .await
            .unwrap();
        mem.load("c").await.unwrap();
        assert_eq!(mem.tracked_conversations(), 1);
        assert_eq!(hook.calls(), 1);

        // After forgetting, the next load on the same (still-populated)
        // backend re-delivers the demotion. This is the documented
        // contract: forget()/restart re-fire the hook, hooks must be
        // idempotent.
        mem.forget("c");
        assert_eq!(mem.tracked_conversations(), 0);
        mem.load("c").await.unwrap();
        assert_eq!(hook.calls(), 2);
    }

    // ----------------------------------------------------------------
    // CompactingMemory tests
    // ----------------------------------------------------------------

    #[tokio::test]
    async fn compacting_no_demotion_returns_kept_only() {
        let mem = CompactingMemory::new(
            InMemoryConversationMemory::new(),
            SlidingWindowMemory::last_messages(10),
            TemplateCompactor::new(),
        );

        mem.append("c", vec![user("hi"), assistant("hello")])
            .await
            .unwrap();
        let loaded = mem.load("c").await.unwrap();
        assert_eq!(loaded.len(), 2);
        // No tracking entry needed when nothing was demoted on the first load.
        // (We may have inserted a default entry; what matters is that no
        // summary message was spliced in.)
        assert!(matches!(&loaded[0], Message::User { .. }));
    }

    #[tokio::test]
    async fn compacting_splices_summary_when_demoted() {
        let mem = CompactingMemory::new(
            InMemoryConversationMemory::new(),
            SlidingWindowMemory::last_messages(2),
            TemplateCompactor::new(),
        );

        mem.append(
            "c",
            vec![
                user("first"),
                assistant("second"),
                user("third"),
                assistant("fourth"),
            ],
        )
        .await
        .unwrap();

        let loaded = mem.load("c").await.unwrap();
        // Expected shape: [summary, third, fourth]
        assert_eq!(loaded.len(), 3);
        let Message::System { content } = &loaded[0] else {
            panic!("expected summary as system message");
        };
        assert!(content.contains("[Conversation summary so far]"));
        assert!(content.contains("user: first"));
        assert!(content.contains("assistant: second"));
        // The kept window is intact.
        let Message::User { content } = &loaded[1] else {
            panic!("expected kept user message");
        };
        let UserContent::Text(t) = content.first_ref() else {
            panic!("expected text");
        };
        assert_eq!(t.text, "third");
    }

    #[tokio::test]
    async fn compacting_rolls_summary_forward() {
        let mem = CompactingMemory::new(
            InMemoryConversationMemory::new(),
            SlidingWindowMemory::last_messages(2),
            TemplateCompactor::new(),
        );

        mem.append(
            "c",
            vec![user("a"), assistant("b"), user("c"), assistant("d")],
        )
        .await
        .unwrap();

        let first = mem.load("c").await.unwrap();
        let Message::System { content } = &first[0] else {
            panic!("summary missing");
        };
        let first_summary = content.clone();
        assert!(first_summary.contains("user: a"));
        assert!(first_summary.contains("assistant: b"));

        // Append more turns; the next load should fold the previous summary
        // into a new one that also covers the newly-evicted prefix.
        mem.append("c", vec![user("e"), assistant("f")])
            .await
            .unwrap();
        let second = mem.load("c").await.unwrap();
        let Message::System { content } = &second[0] else {
            panic!("summary missing");
        };
        // The new summary contains the old summary text (carry_over) plus
        // the freshly-evicted lines.
        assert!(content.contains(&first_summary));
        assert!(content.contains("user: c"));
        assert!(content.contains("assistant: d"));
    }

    #[tokio::test]
    async fn compacting_idempotent_within_process() {
        // Loading twice with no new evictions reuses the stored summary
        // and does not re-run the compactor (we observe this via the
        // produced text: a re-run with a non-None carry_over would double
        // the header line).
        let mem = CompactingMemory::new(
            InMemoryConversationMemory::new(),
            SlidingWindowMemory::last_messages(1),
            TemplateCompactor::new(),
        );
        mem.append("c", vec![user("a"), assistant("b"), user("c")])
            .await
            .unwrap();

        let first = mem.load("c").await.unwrap();
        let second = mem.load("c").await.unwrap();
        assert_eq!(first.len(), second.len());
        let Message::System { content: c1 } = &first[0] else {
            panic!()
        };
        let Message::System { content: c2 } = &second[0] else {
            panic!()
        };
        assert_eq!(c1, c2);
    }

    #[tokio::test]
    async fn compacting_clear_drops_summary() {
        let mem = CompactingMemory::new(
            InMemoryConversationMemory::new(),
            SlidingWindowMemory::last_messages(1),
            TemplateCompactor::new(),
        );
        mem.append("c", vec![user("a"), assistant("b"), user("c")])
            .await
            .unwrap();
        mem.load("c").await.unwrap();
        assert_eq!(mem.tracked_conversations(), 1);

        mem.clear("c").await.unwrap();
        assert_eq!(mem.tracked_conversations(), 0);
        assert!(mem.load("c").await.unwrap().is_empty());
    }

    // A compactor that fails the first call and succeeds afterwards, so we
    // can verify failure is propagated and the watermark is not advanced.
    #[derive(Default)]
    struct FlakyCompactor {
        calls: std::sync::atomic::AtomicUsize,
    }

    impl Compactor for FlakyCompactor {
        type Artifact = TextSummary;

        fn compact<'a>(
            &'a self,
            _conversation_id: &'a str,
            evicted: &'a [Message],
            _carry_over: Option<&'a Self::Artifact>,
        ) -> WasmBoxedFuture<'a, Result<Self::Artifact, MemoryError>> {
            Box::pin(async move {
                let n = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if n == 0 {
                    Err(MemoryError::Policy("flaky".into()))
                } else {
                    Ok(TextSummary(format!("compacted {} messages", evicted.len())))
                }
            })
        }
    }

    #[tokio::test]
    async fn compacting_failure_does_not_advance_watermark() {
        let mem = CompactingMemory::new(
            InMemoryConversationMemory::new(),
            SlidingWindowMemory::last_messages(1),
            FlakyCompactor::default(),
        );
        mem.append("c", vec![user("a"), assistant("b"), user("c")])
            .await
            .unwrap();

        let err = mem.load("c").await.unwrap_err();
        assert!(matches!(err, MemoryError::Policy(_)));

        // Retry should succeed and produce a summary.
        let loaded = mem.load("c").await.unwrap();
        assert_eq!(loaded.len(), 2);
        let Message::System { content } = &loaded[0] else {
            panic!("expected summary")
        };
        assert!(content.contains("compacted"));
    }

    // A compactor that records every invocation, including the lengths of
    // its `evicted` slice and whether `carry_over` was supplied.
    #[derive(Default)]
    struct CountingCompactor {
        log: Mutex<Vec<(usize, bool)>>,
    }

    impl CountingCompactor {
        fn calls(&self) -> Vec<(usize, bool)> {
            self.log.lock().unwrap().clone()
        }
    }

    impl Compactor for CountingCompactor {
        type Artifact = TextSummary;

        fn compact<'a>(
            &'a self,
            _conversation_id: &'a str,
            evicted: &'a [Message],
            carry_over: Option<&'a Self::Artifact>,
        ) -> WasmBoxedFuture<'a, Result<Self::Artifact, MemoryError>> {
            Box::pin(async move {
                self.log
                    .lock()
                    .unwrap()
                    .push((evicted.len(), carry_over.is_some()));
                let prev = carry_over.map(|s| s.as_str()).unwrap_or("");
                Ok(TextSummary(format!("{prev}|{}", evicted.len())))
            })
        }
    }

    #[tokio::test]
    async fn compacting_no_demotion_does_not_invoke_compactor() {
        let compactor = Arc::new(CountingCompactor::default());
        let mem = CompactingMemory::new(
            InMemoryConversationMemory::new(),
            SlidingWindowMemory::last_messages(10),
            compactor.clone(),
        );

        mem.append("c", vec![user("a"), assistant("b")])
            .await
            .unwrap();
        mem.load("c").await.unwrap();
        mem.load("c").await.unwrap();
        mem.load("c").await.unwrap();
        assert!(compactor.calls().is_empty());
        // Fast path means we never installed a tracking entry either.
        assert_eq!(mem.tracked_conversations(), 0);
    }

    #[tokio::test]
    async fn compacting_invokes_compactor_only_on_new_demotions() {
        let compactor = Arc::new(CountingCompactor::default());
        let mem = CompactingMemory::new(
            InMemoryConversationMemory::new(),
            SlidingWindowMemory::last_messages(2),
            compactor.clone(),
        );

        // First eviction: 2 messages demoted.
        mem.append(
            "c",
            vec![user("a"), assistant("b"), user("c"), assistant("d")],
        )
        .await
        .unwrap();
        mem.load("c").await.unwrap();
        // Re-load: nothing new evicted; compactor must NOT run again.
        mem.load("c").await.unwrap();
        mem.load("c").await.unwrap();
        let calls = compactor.calls();
        assert_eq!(
            calls.len(),
            1,
            "compactor invoked more than once: {calls:?}"
        );
        assert_eq!(calls[0], (2, false));

        // Append two more turns → another 2 demoted; compactor runs once
        // more, and this time `carry_over` must be present.
        mem.append("c", vec![user("e"), assistant("f")])
            .await
            .unwrap();
        mem.load("c").await.unwrap();
        mem.load("c").await.unwrap();
        let calls = compactor.calls();
        assert_eq!(calls.len(), 2, "expected exactly one new call: {calls:?}");
        // Second call only compacts the *newly* evicted prefix (2 msgs)
        // with the previous summary as carry-over.
        assert_eq!(calls[1], (2, true));
    }

    #[tokio::test]
    async fn compacting_serialises_concurrent_loads() {
        // Many concurrent loads on the same conversation must produce at
        // most ONE compactor invocation per "epoch" of new evictions.
        let compactor = Arc::new(CountingCompactor::default());
        let mem = Arc::new(CompactingMemory::new(
            InMemoryConversationMemory::new(),
            SlidingWindowMemory::last_messages(2),
            compactor.clone(),
        ));
        mem.append(
            "c",
            vec![user("a"), assistant("b"), user("c"), assistant("d")],
        )
        .await
        .unwrap();

        let mut handles = Vec::new();
        for _ in 0..32 {
            let mem = mem.clone();
            handles.push(tokio::spawn(async move {
                mem.load("c").await.unwrap();
            }));
        }
        for h in handles {
            h.await.unwrap();
        }

        // Exactly one invocation: the first to acquire the lock runs the
        // compactor; the others see in_flight or the advanced watermark.
        let calls = compactor.calls();
        assert_eq!(calls.len(), 1, "expected exactly 1 call: {calls:?}");
    }

    #[tokio::test]
    async fn compacting_clear_drops_summary_carry_over() {
        // After clear, the next load on a freshly-populated backend must
        // start compaction from scratch (carry_over=None), not roll the
        // old summary forward.
        let compactor = Arc::new(CountingCompactor::default());
        let mem = CompactingMemory::new(
            InMemoryConversationMemory::new(),
            SlidingWindowMemory::last_messages(1),
            compactor.clone(),
        );
        mem.append("c", vec![user("a"), assistant("b"), user("c")])
            .await
            .unwrap();
        mem.load("c").await.unwrap();
        assert_eq!(compactor.calls()[0], (2, false));

        mem.clear("c").await.unwrap();
        assert_eq!(mem.tracked_conversations(), 0);

        mem.append("c", vec![user("x"), assistant("y"), user("z")])
            .await
            .unwrap();
        mem.load("c").await.unwrap();
        let calls = compactor.calls();
        assert_eq!(calls.len(), 2);
        // Crucial: no carry_over after clear.
        assert_eq!(calls[1], (2, false));
    }

    #[tokio::test]
    async fn compacting_forget_drops_summary() {
        let compactor = Arc::new(CountingCompactor::default());
        let mem = CompactingMemory::new(
            InMemoryConversationMemory::new(),
            SlidingWindowMemory::last_messages(1),
            compactor.clone(),
        );
        mem.append("c", vec![user("a"), assistant("b"), user("c")])
            .await
            .unwrap();
        mem.load("c").await.unwrap();
        assert_eq!(mem.tracked_conversations(), 1);
        mem.forget("c");
        assert_eq!(mem.tracked_conversations(), 0);

        // Next load on the still-populated backend re-compacts from
        // scratch — same documented contract as DemotionHook.
        mem.load("c").await.unwrap();
        let calls = compactor.calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[1], (2, false));
    }

    #[tokio::test]
    async fn compacting_arc_compactor_works() {
        // Arc<C> forwarding impl exists on Compactor, so CompactingMemory
        // must accept it.
        let compactor: Arc<dyn Compactor<Artifact = TextSummary>> =
            Arc::new(TemplateCompactor::new());
        let mem = CompactingMemory::new(
            InMemoryConversationMemory::new(),
            SlidingWindowMemory::last_messages(1),
            compactor,
        );
        mem.append("c", vec![user("a"), assistant("b"), user("c")])
            .await
            .unwrap();
        let loaded = mem.load("c").await.unwrap();
        assert_eq!(loaded.len(), 2);
        assert!(matches!(&loaded[0], Message::System { .. }));
    }

    #[tokio::test]
    async fn compacting_into_inner_returns_components() {
        let mem = CompactingMemory::new(
            InMemoryConversationMemory::new(),
            SlidingWindowMemory::last_messages(1),
            TemplateCompactor::new(),
        );
        let (_inner, _policy, _compactor) = mem.into_inner();
    }

    #[tokio::test]
    async fn compacting_isolates_conversations() {
        let compactor = Arc::new(CountingCompactor::default());
        let mem = CompactingMemory::new(
            InMemoryConversationMemory::new(),
            SlidingWindowMemory::last_messages(1),
            compactor.clone(),
        );
        mem.append("a", vec![user("a1"), assistant("a2"), user("a3")])
            .await
            .unwrap();
        mem.append("b", vec![user("b1"), assistant("b2"), user("b3")])
            .await
            .unwrap();

        let a = mem.load("a").await.unwrap();
        let b = mem.load("b").await.unwrap();
        // Each conversation gets its own summary.
        assert_eq!(a.len(), 2);
        assert_eq!(b.len(), 2);
        assert_eq!(compactor.calls().len(), 2);
        assert_eq!(mem.tracked_conversations(), 2);
    }

    #[tokio::test]
    async fn compacting_composes_with_token_window() {
        // Verify CompactingMemory is policy-agnostic: works over a
        // TokenWindowMemory just as well as a SlidingWindowMemory.
        let mem = CompactingMemory::new(
            InMemoryConversationMemory::new(),
            TokenWindowMemory::new(30, HeuristicTokenCounter::openai()),
            TemplateCompactor::new(),
        );
        mem.append(
            "c",
            vec![
                user("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
                assistant("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"),
                user("cccccccccccccccccccc"),
                assistant("d"),
            ],
        )
        .await
        .unwrap();
        let loaded = mem.load("c").await.unwrap();
        // Some prefix should have been evicted; expect a summary in front.
        assert!(loaded.len() >= 2);
        assert!(matches!(&loaded[0], Message::System { .. }));
    }

    #[tokio::test]
    async fn template_compactor_renders_system_messages() {
        let compactor = TemplateCompactor::new();
        let evicted = vec![
            Message::System {
                content: "you are helpful".into(),
            },
            user("hi"),
            assistant("hello"),
        ];
        let summary = compactor.compact("c", &evicted, None).await.unwrap();
        let s = summary.as_str();
        assert!(s.contains("system: you are helpful"), "got: {s}");
        assert!(s.contains("user: hi"));
        assert!(s.contains("assistant: hello"));
    }

    #[tokio::test]
    async fn template_compactor_renders_tool_call_marker() {
        let compactor = TemplateCompactor::new();
        let evicted = vec![tool_call_msg(), tool_result_msg()];
        let summary = compactor.compact("c", &evicted, None).await.unwrap();
        let s = summary.as_str();
        assert!(s.contains("[tool call: t]"), "got: {s}");
        assert!(s.contains("[tool result]"), "got: {s}");
    }

    #[tokio::test]
    async fn template_compactor_carry_over_threaded() {
        let compactor = TemplateCompactor::new();
        let first = compactor
            .compact("c", &[user("hello")], None)
            .await
            .unwrap();
        assert!(!first.as_str().is_empty());

        let second = compactor
            .compact("c", &[assistant("world")], Some(&first))
            .await
            .unwrap();
        // Carry-over text appears in the new summary.
        assert!(second.as_str().contains(first.as_str()));
        assert!(second.as_str().contains("assistant: world"));
    }

    #[tokio::test]
    async fn template_compactor_artifact_into_message() {
        let s = TextSummary("rolled-up text".into());
        let msg: Message = s.into();
        let Message::System { content } = msg else {
            panic!("expected system message");
        };
        assert_eq!(content, "rolled-up text");
    }

    #[tokio::test]
    async fn template_compactor_caps_summary_at_max_bytes() {
        let cap = 256;
        let compactor = TemplateCompactor::new().with_max_bytes(cap);
        // Build an evicted history large enough to exceed `cap` on its own.
        let mut evicted = Vec::new();
        for i in 0..50 {
            evicted.push(user(&format!("message number {i} with some filler")));
        }
        let summary = compactor.compact("c", &evicted, None).await.unwrap();
        assert!(
            summary.as_str().len()
                <= cap + "[Conversation summary so far]\n[\u{2026}truncated\u{2026}]\n".len(),
            "summary len {} exceeds cap {} (plus header+marker)",
            summary.as_str().len(),
            cap,
        );
        // Header is preserved.
        assert!(
            summary
                .as_str()
                .starts_with("[Conversation summary so far]\n")
        );
        // Truncation marker is present.
        assert!(summary.as_str().contains("[\u{2026}truncated\u{2026}]"));
        // Most recent line survives.
        assert!(summary.as_str().contains("message number 49"));
    }

    #[tokio::test]
    async fn template_compactor_unbounded_by_default() {
        let compactor = TemplateCompactor::new();
        let mut evicted = Vec::new();
        for i in 0..200 {
            evicted.push(user(&format!("msg {i}")));
        }
        let summary = compactor.compact("c", &evicted, None).await.unwrap();
        // Without a cap, no truncation marker should appear.
        assert!(!summary.as_str().contains("[\u{2026}truncated\u{2026}]"));
        // Both ends are present.
        assert!(summary.as_str().contains("msg 0"));
        assert!(summary.as_str().contains("msg 199"));
    }

    #[tokio::test]
    async fn template_compactor_with_max_bytes_zero_is_unbounded() {
        let compactor = TemplateCompactor::new().with_max_bytes(0);
        let mut evicted = Vec::new();
        for i in 0..200 {
            evicted.push(user(&format!("msg {i}")));
        }
        let summary = compactor.compact("c", &evicted, None).await.unwrap();
        assert!(!summary.as_str().contains("[\u{2026}truncated\u{2026}]"));
    }

    #[tokio::test]
    async fn compacting_summary_stays_bounded_across_rolls() {
        // With a capped TemplateCompactor, repeated rolling must not let
        // the summary grow without bound.
        let cap = 512;
        let mem = CompactingMemory::new(
            InMemoryConversationMemory::new(),
            SlidingWindowMemory::last_messages(2),
            TemplateCompactor::new().with_max_bytes(cap),
        );
        mem.append("c", vec![user("seed-a"), assistant("seed-b")])
            .await
            .unwrap();
        for i in 0..30 {
            mem.append(
                "c",
                vec![
                    user(&format!("user line {i} ----- padding padding padding")),
                    assistant(&format!("assistant line {i} ----- padding padding")),
                ],
            )
            .await
            .unwrap();
            mem.load("c").await.unwrap();
        }
        let loaded = mem.load("c").await.unwrap();
        let Message::System { content } = &loaded[0] else {
            panic!("expected summary");
        };
        // Allow some slack for header + marker overhead.
        let slack = "[Conversation summary so far]\n[\u{2026}truncated\u{2026}]\n".len();
        assert!(
            content.len() <= cap + slack,
            "summary grew to {} bytes (cap {}, slack {})",
            content.len(),
            cap,
            slack,
        );
    }

    #[tokio::test]
    async fn compacting_concurrent_with_clear_does_not_resurrect_state() {
        // A clear that lands while compaction is in flight must not be
        // overwritten by the post-await state update.
        use std::sync::atomic::{AtomicBool, Ordering};

        struct GatedCompactor {
            release: tokio::sync::Notify,
            entered: AtomicBool,
        }

        impl Compactor for GatedCompactor {
            type Artifact = TextSummary;

            fn compact<'a>(
                &'a self,
                _conversation_id: &'a str,
                _evicted: &'a [Message],
                _carry_over: Option<&'a Self::Artifact>,
            ) -> WasmBoxedFuture<'a, Result<Self::Artifact, MemoryError>> {
                Box::pin(async move {
                    self.entered.store(true, Ordering::SeqCst);
                    self.release.notified().await;
                    Ok(TextSummary("late summary".into()))
                })
            }
        }

        let compactor = Arc::new(GatedCompactor {
            release: tokio::sync::Notify::new(),
            entered: AtomicBool::new(false),
        });
        let mem = Arc::new(CompactingMemory::new(
            InMemoryConversationMemory::new(),
            SlidingWindowMemory::last_messages(1),
            compactor.clone(),
        ));
        mem.append("c", vec![user("a"), assistant("b"), user("c")])
            .await
            .unwrap();

        // Kick off a load that will block inside the compactor.
        let mem_load = mem.clone();
        let load_handle = tokio::spawn(async move { mem_load.load("c").await });

        // Wait for the compactor to have entered.
        while !compactor.entered.load(Ordering::SeqCst) {
            tokio::task::yield_now().await;
        }

        // Clear while the compaction is in flight.
        mem.clear("c").await.unwrap();

        // Release the compactor; it should complete and *not* resurrect
        // the cleared state.
        compactor.release.notify_one();
        let _ = load_handle.await.unwrap();

        assert_eq!(mem.tracked_conversations(), 0);
        // A subsequent load on the empty backend returns nothing.
        assert!(mem.load("c").await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn compacting_dropped_load_releases_in_flight_gate() {
        // If a `load(...)` future is dropped while awaiting the
        // compactor, the in-flight gate must not leak: subsequent loads
        // on the same conversation must be able to retry compaction.
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct GatedCompactor {
            release: tokio::sync::Notify,
            entered: AtomicUsize,
        }

        impl Compactor for GatedCompactor {
            type Artifact = TextSummary;

            fn compact<'a>(
                &'a self,
                _conversation_id: &'a str,
                _evicted: &'a [Message],
                _carry_over: Option<&'a Self::Artifact>,
            ) -> WasmBoxedFuture<'a, Result<Self::Artifact, MemoryError>> {
                Box::pin(async move {
                    self.entered.fetch_add(1, Ordering::SeqCst);
                    self.release.notified().await;
                    Ok(TextSummary("ran".into()))
                })
            }
        }

        let compactor = Arc::new(GatedCompactor {
            release: tokio::sync::Notify::new(),
            entered: AtomicUsize::new(0),
        });
        let mem = Arc::new(CompactingMemory::new(
            InMemoryConversationMemory::new(),
            SlidingWindowMemory::last_messages(1),
            compactor.clone(),
        ));
        mem.append("c", vec![user("a"), assistant("b"), user("c")])
            .await
            .unwrap();

        // Kick off a load that will block inside the compactor, then
        // abort it while awaiting — simulating a caller-side timeout
        // or `tokio::select!` cancellation.
        let mem_load = mem.clone();
        let handle = tokio::spawn(async move { mem_load.load("c").await });
        while compactor.entered.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
        handle.abort();
        let _ = handle.await;

        // The aborted future was dropped without clearing in_flight via
        // the success/error branches; the RAII guard's `Drop` should
        // have released it. A new load must therefore be able to drive
        // a fresh compaction rather than short-circuiting forever.
        let mem_load = mem.clone();
        let retry = tokio::spawn(async move { mem_load.load("c").await });
        // Wait for the compactor to be entered a second time. If the
        // gate had leaked, this would never happen — the load would
        // short-circuit on `in_flight = true` and return immediately.
        while compactor.entered.load(Ordering::SeqCst) < 2 {
            tokio::task::yield_now().await;
        }
        compactor.release.notify_waiters();
        let loaded = retry.await.unwrap().unwrap();
        assert_eq!(loaded.len(), 2);
        let Message::System { content } = &loaded[0] else {
            panic!("expected summary")
        };
        assert_eq!(content, "ran");
    }

    #[tokio::test]
    async fn template_compactor_caps_summary_with_multiline_header() {
        // A header containing embedded newlines must not break the
        // truncation boundary calculation. The first newline in the
        // assembled buffer marks the header/body split, regardless of
        // how the caller chose to format the header.
        let cap = 256;
        let compactor = TemplateCompactor::with_header("line one\nline two").with_max_bytes(cap);
        let mut evicted = Vec::new();
        for i in 0..50 {
            evicted.push(user(&format!("message number {i} with some filler")));
        }
        let summary = compactor.compact("c", &evicted, None).await.unwrap();
        let text = summary.as_str();

        // The first line of the header is preserved as the header line.
        assert!(text.starts_with("line one\n"));
        // Truncation marker is present and the most recent line survives.
        assert!(text.contains("[\u{2026}truncated\u{2026}]"));
        assert!(text.contains("message number 49"));
        // Cap is honoured up to the header+marker overhead.
        let overhead = "line one\n".len() + "[\u{2026}truncated\u{2026}]\n".len();
        assert!(
            text.len() <= cap + overhead,
            "summary len {} exceeds cap {} plus overhead {}",
            text.len(),
            cap,
            overhead,
        );
    }
}
