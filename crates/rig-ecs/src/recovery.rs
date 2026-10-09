//! What a turn does when a model call fails, as rig-core's [`RETRY`]
//! policy decides: wait and call again after a transient failure; clear old
//! tool outputs (rig-memory's [`ClearToolOutputs`]) and call again when the
//! conversation outgrew the model's context window. Each turn counts its attempts in its [`Recovery`]; a
//! wait is a call entity of the turn with a [`Backoff`], so interrupting
//! the turn cancels it like any other call.

use std::time::Duration;

use bevy_ecs::prelude::*;
use rig_core::error::retry::RetryPolicy;
use rig_memory::ClearToolOutputs;
use web_time::Instant;

/// How failed model calls are retried: rig-core's default, four retries in
/// a row per turn (a reply resets the count).
pub const RETRY: RetryPolicy = RetryPolicy::DEFAULT;
/// Tokens of the newest tool outputs a clearing keeps (opencode's
/// `PRUNE_PROTECT`).
const KEEP_RECENT_OUTPUTS: usize = 40_000;

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

/// How old tool outputs are cleared: all but the newest 40k tokens of them.
pub fn clearing() -> ClearToolOutputs {
    ClearToolOutputs::new(KEEP_RECENT_OUTPUTS)
}
