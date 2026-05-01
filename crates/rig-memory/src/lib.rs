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

use std::sync::Arc;

/// Re-exports of the core memory abstractions so callers only need a single
/// dependency on `rig-memory` for both the trait/backend and the policies.
pub use rig_core::memory::{ConversationMemory, InMemoryConversationMemory, MemoryError};

use rig_core::completion::Message;
use rig_core::message::UserContent;
use rig_core::wasm_compat::{WasmCompatSend, WasmCompatSync};

/// A transformation applied to messages loaded from a [`ConversationMemory`].
///
/// Policies typically truncate, summarize, or re-order history. They are
/// pure, fallible message transformers: implementors that cannot fail should
/// always return `Ok`.
pub trait MemoryPolicy: WasmCompatSend + WasmCompatSync {
    /// Transform `messages` into the history that should be returned to the agent.
    fn apply(&self, messages: Vec<Message>) -> Result<Vec<Message>, MemoryError>;
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
        if messages.len() <= self.max_messages {
            return Ok(messages);
        }

        let start = messages.len() - self.max_messages;
        let mut window: Vec<Message> = messages.into_iter().skip(start).collect();

        drop_leading_orphan_tool_result(&mut window);
        Ok(window)
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

        let mut window: Vec<Message> = messages.into_iter().skip(keep_from).collect();
        drop_leading_orphan_tool_result(&mut window);
        Ok(window)
    }
}

fn drop_leading_orphan_tool_result(window: &mut Vec<Message>) {
    if let Some(Message::User { content }) = window.first()
        && matches!(content.first(), UserContent::ToolResult(_))
    {
        window.remove(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rig_core::OneOrMany;
    use rig_core::message::{
        AssistantContent, ToolCall, ToolFunction, ToolResult, ToolResultContent, UserContent,
    };

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
}
