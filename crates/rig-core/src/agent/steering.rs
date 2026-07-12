//! Run-scoped steering for active agent prompt loops.
//!
//! A [`SteeringHandle`] belongs to one [`AgentRunner`](super::AgentRunner). Messages
//! submitted through it are delivered at the next safe boundary: after the
//! current model turn and its tool batch have settled, immediately before the
//! next model call. Handles are deliberately run-scoped so concurrent runs,
//! including runs sharing a conversation ID, cannot consume each other's input.

use std::{
    collections::VecDeque,
    sync::{Arc, Mutex, MutexGuard},
};

use crate::completion::Message;

/// Error returned when steering input can no longer be accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum SteeringError {
    /// The associated request has settled or was dropped.
    #[error("the agent run is no longer accepting steering messages")]
    RunClosed,
}

#[derive(Debug, Default)]
struct SteeringState {
    pending: VecDeque<Message>,
    closed: bool,
}

/// A cloneable, run-scoped handle for steering an active agent.
///
/// Obtain one from [`AgentRunner::steering_handle`](super::AgentRunner::steering_handle)
/// or from a prompt request's `steering_handle` method before moving that request
/// into another task. Calling [`steer`](Self::steer) queues a user message; it
/// does not interrupt an in-flight model request or tool. The shared agent driver
/// drains queued messages after the current tool batch and before the next model
/// call. If the current assistant turn would otherwise finish the run, queued
/// steering starts another model turn, subject to the run's `max_turns` budget.
///
/// The handle is tied to one request, not a conversation ID. Cloning an [`Agent`](super::Agent)
/// and starting another request creates a different steering queue. The queue is
/// intentionally unbounded; callers control how much input they enqueue before
/// the run reaches its next safe boundary.
#[derive(Clone)]
pub struct SteeringHandle {
    inner: Arc<Mutex<SteeringState>>,
}

impl std::fmt::Debug for SteeringHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = lock_state(&self.inner);
        f.debug_struct("SteeringHandle")
            .field("pending", &state.pending.len())
            .field("closed", &state.closed)
            .finish()
    }
}

impl SteeringHandle {
    /// Queue a user message for the next safe boundary in this run.
    ///
    /// Messages are delivered in submission order. Returns
    /// [`SteeringError::RunClosed`] once the run has settled or its request has
    /// been dropped, ensuring late input cannot leak into a later run.
    pub fn steer(&self, message: impl Into<String>) -> Result<(), SteeringError> {
        let mut state = lock_state(&self.inner);
        if state.closed {
            return Err(SteeringError::RunClosed);
        }
        state.pending.push_back(Message::user(message));
        Ok(())
    }

    /// Whether the associated run has stopped accepting steering input.
    pub fn is_closed(&self) -> bool {
        lock_state(&self.inner).closed
    }
}

/// Driver-owned half of a steering queue.
///
/// Unlike [`SteeringHandle`], this type is not cloneable. Dropping the request
/// owner closes the queue even when public handles remain alive.
pub(crate) struct SteeringInbox {
    inner: Arc<Mutex<SteeringState>>,
}

impl std::fmt::Debug for SteeringInbox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = lock_state(&self.inner);
        f.debug_struct("SteeringInbox")
            .field("pending", &state.pending.len())
            .field("closed", &state.closed)
            .finish()
    }
}

impl SteeringInbox {
    pub(crate) fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(SteeringState::default())),
        }
    }

    pub(crate) fn handle(&self) -> SteeringHandle {
        SteeringHandle {
            inner: self.inner.clone(),
        }
    }

    /// Drain input without closing the run. Used immediately before a model call.
    pub(crate) fn drain(&self) -> Vec<Message> {
        lock_state(&self.inner).pending.drain(..).collect()
    }

    /// Atomically drain pending input or close an idle queue.
    ///
    /// `Some(messages)` means the run must continue. `None` means no steering
    /// was pending and the queue is now closed, so a concurrent late sender
    /// cannot race with finalization and leave input stranded.
    pub(crate) fn drain_or_close(&self) -> Option<Vec<Message>> {
        let mut state = lock_state(&self.inner);
        if state.pending.is_empty() {
            state.closed = true;
            None
        } else {
            Some(state.pending.drain(..).collect())
        }
    }

    fn close(&self) {
        let mut state = lock_state(&self.inner);
        state.closed = true;
        state.pending.clear();
    }
}

impl Drop for SteeringInbox {
    fn drop(&mut self) {
        self.close();
    }
}

fn lock_state(state: &Mutex<SteeringState>) -> MutexGuard<'_, SteeringState> {
    match state.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closing_an_idle_inbox_rejects_late_input() {
        let inbox = SteeringInbox::new();
        let handle = inbox.handle();

        assert!(inbox.drain_or_close().is_none());
        assert_eq!(handle.steer("too late"), Err(SteeringError::RunClosed));
        assert!(handle.is_closed());
    }

    #[test]
    fn debug_does_not_expose_queued_messages() {
        let inbox = SteeringInbox::new();
        let handle = inbox.handle();
        handle.steer("secret steering input").unwrap();

        let debug = format!("{handle:?}");
        assert!(debug.contains("pending: 1"));
        assert!(!debug.contains("secret steering input"));
    }

    #[test]
    fn dropping_the_request_owner_closes_the_handle() {
        let inbox = SteeringInbox::new();
        let handle = inbox.handle();
        drop(inbox);

        assert_eq!(handle.steer("too late"), Err(SteeringError::RunClosed));
    }
}
