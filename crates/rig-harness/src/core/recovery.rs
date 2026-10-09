//! What a turn does when a model call fails, as rig-core's [`RETRY`]
//! policy decides: wait and call again after a transient failure; clear old
//! tool outputs and call again when the conversation outgrew the model's
//! context window. Each turn counts its attempts in its [`Recovery`]; a
//! wait is a call entity of the turn with a [`Backoff`], so interrupting
//! the turn cancels it like any other call.

use std::time::{Duration, Instant};

use bevy_ecs::prelude::*;
use rig_core::completion::Message;
use rig_core::error::retry::RetryPolicy;
use rig_core::message::{ToolResultContent, UserContent};

use super::compaction::estimate_content;

/// How failed model calls are retried: rig-core's default, four retries in
/// a row per turn (a reply resets the count).
pub const RETRY: RetryPolicy = RetryPolicy::DEFAULT;
/// Tokens of the newest tool outputs a clearing keeps (opencode's
/// `PRUNE_PROTECT`).
pub const KEEP_RECENT_OUTPUTS: u64 = 40_000;
/// What a cleared tool output says instead.
pub const CLEARED: &str =
    "[output cleared to fit the context window; run the tool again if needed]";

/// A turn's recovery so far: the failed calls retried since its last
/// reply, whether it cleared tool outputs after an overflow, and how often
/// it compacted the conversation.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct Recovery {
    /// Retries since the last reply.
    pub retries: u32,
    /// Whether an overflow cleared tool outputs in this turn.
    pub cleared: bool,
    /// Compactions in this turn.
    pub compactions: u32,
}

/// A wait before the turn's next model call, on a call entity of the turn;
/// its task is a [`Running<RetryDue>`](super::calls::Running).
#[derive(Component, Clone, Debug)]
pub struct Backoff {
    /// Which retry this wait is for, from 1 to [`RETRY`]'s `max_retries`.
    pub attempt: u32,
    /// When the call is sent again.
    pub until: Instant,
    /// Why the last call failed.
    pub why: String,
}

impl Backoff {
    /// Whole seconds left to wait, rounded up.
    pub fn seconds_left(&self) -> u64 {
        let left = self.until.saturating_duration_since(Instant::now());
        left.as_secs() + u64::from(left.subsec_nanos() > 0)
    }
}

/// What a [`Backoff`]'s task returns once its wait is over.
pub struct RetryDue;

/// Waits `delay`, on futures-timer's timer thread: no pool thread sleeps,
/// and dropping the task stops the wait.
pub(crate) async fn wait(delay: Duration) -> RetryDue {
    futures_timer::Delay::new(delay).await;
    RetryDue
}

/// What a clearing took out.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cleared {
    /// Tool results cleared.
    pub results: usize,
    /// Their estimated tokens.
    pub tokens: u64,
}

/// Clears the outputs of older tool calls in `messages` to free context,
/// newest kept first: walking back from the end, the outputs within the
/// first `keep` estimated tokens stay, and every older one is replaced by
/// [`CLEARED`]. The last message is never touched: it is what the model
/// must answer. Error results stay, being short and telling the model what
/// went wrong. The calls themselves stay, so the conversation keeps its
/// shape (opencode's `prune`,
/// `references/opencode/packages/opencode/src/session/compaction.ts:271-316`).
pub fn clear_tool_outputs(messages: &mut [Message], keep: u64) -> Cleared {
    let mut cleared = Cleared::default();
    let mut kept = 0u64;
    let Some((_, earlier)) = messages.split_last_mut() else {
        return cleared;
    };
    for message in earlier.iter_mut().rev() {
        let Message::User { content } = message else {
            continue;
        };
        for item in content.iter_mut().rev() {
            let tokens = estimate_content(item);
            let UserContent::ToolResult(result) = item else {
                continue;
            };
            if result.is_error || is_cleared(&result.content) {
                continue;
            }
            kept += tokens;
            if kept <= keep {
                continue;
            }
            result.content = vec![ToolResultContent::text(CLEARED)];
            cleared.results += 1;
            cleared.tokens += tokens;
        }
    }
    cleared
}

fn is_cleared(content: &[ToolResultContent]) -> bool {
    matches!(content, [only] if only.as_text() == Some(CLEARED))
}
