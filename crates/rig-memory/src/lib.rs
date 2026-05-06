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
    ConversationMemory, DemotionHook, InMemoryConversationMemory, MemoryError, NoopDemotionHook,
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
    /// Transform `messages` and report which messages were demoted (excluded
    /// from the returned history).
    ///
    /// Returns `(kept, demoted)`. Truncating policies (sliding window, token
    /// window, …) populate `demoted` with the evicted prefix; non-truncating
    /// policies (e.g. [`NoopMemoryPolicy`]) return an empty `demoted` list.
    ///
    /// Implementors must guarantee that `kept ⊆ messages` in their
    /// iteration order, otherwise [`DemotingPolicyMemory`] may return
    /// inconsistent history. Order-preserving truncation policies satisfy
    /// this trivially.
    ///
    /// This is the canonical method — [`MemoryPolicy::apply`] is a thin
    /// wrapper that discards the demoted half. Implementors only override
    /// this method.
    fn apply_with_demoted(
        &self,
        messages: Vec<Message>,
    ) -> Result<(Vec<Message>, Vec<Message>), MemoryError>;

    /// Transform `messages` into the history that should be returned to the
    /// agent. Equivalent to discarding the demoted half of
    /// [`MemoryPolicy::apply_with_demoted`].
    fn apply(&self, messages: Vec<Message>) -> Result<Vec<Message>, MemoryError> {
        Ok(self.apply_with_demoted(messages)?.0)
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
    fn apply_with_demoted(
        &self,
        messages: Vec<Message>,
    ) -> Result<(Vec<Message>, Vec<Message>), MemoryError> {
        Ok((messages, Vec::new()))
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
            && matches!(content.first(), UserContent::ToolResult(_))
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
/// character lengths.
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
/// the counter sums character lengths and divides by `chars_per_token`,
/// rounded up. Tool calls are charged the JSON-serialised length of their
/// `ToolFunction` payload. Each message is charged a flat
/// `per_message_overhead` to model the per-turn role/separator tokens that
/// providers add internally. Non-text blocks (images, audio, video,
/// documents) are charged `per_attachment_tokens` each because their real
/// cost is provider-specific and rarely text-derived.
///
/// # Presets
///
/// The defaults match OpenAI's published rule of thumb (~4 chars per token,
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
    chars_per_token: f32,
    per_message_overhead: usize,
    per_attachment_tokens: usize,
}

impl HeuristicTokenCounter {
    /// Create a counter with explicit parameters.
    ///
    /// `chars_per_token` is clamped to a minimum of `1.0` so the counter
    /// never panics or produces zero-cost messages on degenerate input.
    pub fn new(
        chars_per_token: f32,
        per_message_overhead: usize,
        per_attachment_tokens: usize,
    ) -> Self {
        let chars_per_token = if chars_per_token.is_finite() && chars_per_token >= 1.0 {
            chars_per_token
        } else {
            1.0
        };
        Self {
            chars_per_token,
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

    fn chars_to_tokens(&self, chars: usize) -> usize {
        // `chars_per_token` is clamped to >= 1.0 in the constructor, so the
        // division is well-defined. We round up so a single non-empty
        // character still costs at least one token.
        let tokens = (chars as f32) / self.chars_per_token;
        tokens.ceil() as usize
    }

    fn count_user(&self, content: &rig_core::message::UserContent) -> usize {
        use rig_core::message::UserContent;
        match content {
            UserContent::Text(text) => self.chars_to_tokens(text.text.chars().count()),
            UserContent::ToolResult(result) => result
                .content
                .iter()
                .map(|c| match c {
                    rig_core::message::ToolResultContent::Text(t) => {
                        self.chars_to_tokens(t.text.chars().count())
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
            AssistantContent::Text(text) => self.chars_to_tokens(text.text.chars().count()),
            AssistantContent::Reasoning(reasoning) => {
                self.chars_to_tokens(reasoning.display_text().chars().count())
            }
            AssistantContent::ToolCall(call) => {
                let name_chars = call.function.name.chars().count();
                // `serde_json::Value::to_string` is the canonical compact JSON
                // encoding and never fails, so we charge tool calls by the
                // length of their serialised arguments without pulling in a
                // direct `serde_json` dependency.
                let args_chars = call.function.arguments.to_string().chars().count();
                self.chars_to_tokens(name_chars + args_chars)
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
            Message::System { content } => self.chars_to_tokens(content.chars().count()),
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
            && matches!(content.first(), UserContent::ToolResult(_))
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
    pub fn forget(&self, conversation_id: &str) -> Result<(), MemoryError> {
        let mut guard = self.state.lock().map_err(poisoned)?;
        guard.remove(conversation_id);
        Ok(())
    }

    /// Number of conversations currently tracked in the watermark map.
    /// Useful for telemetry and leak detection.
    pub fn tracked_conversations(&self) -> Result<usize, MemoryError> {
        let guard = self.state.lock().map_err(poisoned)?;
        Ok(guard.len())
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
            let pending = {
                let mut guard = self.state.lock().map_err(poisoned)?;
                let entry = guard.entry(conversation_id.to_string()).or_default();
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
            };

            if pending.is_empty() {
                return Ok(kept);
            }

            let result = self.hook.on_demote(conversation_id, pending).await;

            // Reacquire briefly to advance the watermark on success and
            // always clear the in-flight flag so a future load can retry.
            {
                let mut guard = self.state.lock().map_err(poisoned)?;
                let entry = guard.entry(conversation_id.to_string()).or_default();
                entry.in_flight = false;
                if result.is_ok() {
                    entry.delivered = demoted_count;
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
            self.forget(conversation_id)?;
            Ok(())
        })
    }
}

fn poisoned<E: std::fmt::Display>(err: E) -> MemoryError {
    MemoryError::backend(std::io::Error::other(err.to_string()))
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
    fn heuristic_counter_clamps_invalid_chars_per_token() {
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
            fn apply_with_demoted(
                &self,
                _: Vec<Message>,
            ) -> Result<(Vec<Message>, Vec<Message>), MemoryError> {
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
            fn apply_with_demoted(
                &self,
                _: Vec<Message>,
            ) -> Result<(Vec<Message>, Vec<Message>), MemoryError> {
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
        assert_eq!(mem.tracked_conversations().unwrap(), 1);
        assert_eq!(hook.calls(), 1);

        // After forgetting, the next load on the same (still-populated)
        // backend re-delivers the demotion. This is the documented
        // contract: forget()/restart re-fire the hook, hooks must be
        // idempotent.
        mem.forget("c").unwrap();
        assert_eq!(mem.tracked_conversations().unwrap(), 0);
        mem.load("c").await.unwrap();
        assert_eq!(hook.calls(), 2);
    }
}
