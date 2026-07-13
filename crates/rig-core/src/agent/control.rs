//! Run-scoped identity, cancellation, steering, and lifecycle control.
//!
//! A [`RunControlHandle`] is created with every [`AgentRunner`](super::AgentRunner).
//! Clone it before starting the runner to steer, follow up, pause, resume, or
//! cancel the same run from another task. Control is observed at safe agent-loop
//! boundaries and while model/tool futures are pending.

use std::{
    collections::VecDeque,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU8, Ordering},
    },
    time::{Duration, Instant},
};

use tokio::sync::watch;

use crate::completion::Message;

use super::RunId;

/// Observable lifecycle state of an agent run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RunStatus {
    /// Constructed but not yet driven.
    Pending,
    /// Model or tool work is active.
    Running,
    /// Paused at a safe boundary.
    Paused,
    /// Completed successfully.
    Completed,
    /// Cancelled by the host or deadline.
    Cancelled,
    /// Failed with an error.
    Failed,
    /// Exhausted its configured turn budget.
    Exhausted,
}

impl RunStatus {
    const fn code(self) -> u8 {
        match self {
            Self::Pending => 0,
            Self::Running => 1,
            Self::Paused => 2,
            Self::Completed => 3,
            Self::Cancelled => 4,
            Self::Failed => 5,
            Self::Exhausted => 6,
        }
    }

    const fn from_code(code: u8) -> Self {
        match code {
            1 => Self::Running,
            2 => Self::Paused,
            3 => Self::Completed,
            4 => Self::Cancelled,
            5 => Self::Failed,
            6 => Self::Exhausted,
            _ => Self::Pending,
        }
    }
}

#[derive(Debug, Default)]
struct Queues {
    steer: VecDeque<Message>,
    follow_up: VecDeque<Message>,
    next_turn: VecDeque<Message>,
}

#[derive(Debug)]
struct RunControlInner {
    run_id: RunId,
    status: AtomicU8,
    queues: Mutex<Queues>,
    changed: watch::Sender<u64>,
    deadline: Mutex<Option<Instant>>,
}

/// Cloneable host control for one agent run.
///
/// Queue ownership is run-scoped: steering and follow-up messages are consumed
/// only by the runner that created this handle. `next_turn` messages are never
/// consumed by an active run and can be drained by the host after it settles.
#[derive(Debug, Clone)]
pub struct RunControlHandle(Arc<RunControlInner>);

impl Default for RunControlHandle {
    fn default() -> Self {
        Self::new()
    }
}

impl RunControlHandle {
    /// Create an independent run control handle.
    pub fn new() -> Self {
        Self(Arc::new(RunControlInner {
            run_id: RunId::generate(),
            status: AtomicU8::new(RunStatus::Pending.code()),
            queues: Mutex::new(Queues::default()),
            changed: watch::channel(0).0,
            deadline: Mutex::new(None),
        }))
    }

    /// Stable process-scoped identity for this run.
    pub fn run_id(&self) -> &RunId {
        &self.0.run_id
    }

    /// Current lifecycle state.
    pub fn status(&self) -> RunStatus {
        RunStatus::from_code(self.0.status.load(Ordering::Acquire))
    }

    /// Request cancellation. Repeated calls are idempotent.
    pub fn cancel(&self) {
        if matches!(
            self.status(),
            RunStatus::Completed | RunStatus::Cancelled | RunStatus::Failed | RunStatus::Exhausted
        ) {
            return;
        }
        let mut queues = self.lock_queues();
        queues.steer.clear();
        queues.follow_up.clear();
        drop(queues);
        self.set_status(RunStatus::Cancelled);
    }

    /// Queue a message for the next safe boundary before a model call.
    pub fn steer(&self, message: impl Into<Message>) {
        self.lock_queues().steer.push_back(message.into());
        self.notify();
    }

    /// Queue a message to run only when the active run would otherwise finish.
    pub fn follow_up(&self, message: impl Into<Message>) {
        self.lock_queues().follow_up.push_back(message.into());
        self.notify();
    }

    /// Queue a message for a future host-started turn. Active runs never drain it.
    pub fn next_turn(&self, message: impl Into<Message>) {
        self.lock_queues().next_turn.push_back(message.into());
        self.notify();
    }

    /// Drain messages queued for future host-started turns.
    pub fn drain_next_turn(&self) -> Vec<Message> {
        self.lock_queues().next_turn.drain(..).collect()
    }

    /// Ask the run to pause at its next safe boundary.
    pub fn pause(&self) {
        if !matches!(
            self.status(),
            RunStatus::Completed | RunStatus::Cancelled | RunStatus::Failed | RunStatus::Exhausted
        ) {
            self.set_status(RunStatus::Paused);
        }
    }

    /// Resume a paused run.
    pub fn resume(&self) {
        if self.status() == RunStatus::Paused {
            self.set_status(RunStatus::Running);
        }
    }

    /// Set a relative deadline for the run.
    pub fn deadline_after(&self, duration: Duration) {
        *self.lock_deadline() = Instant::now().checked_add(duration);
        self.notify();
    }

    /// Clear the configured deadline.
    pub fn clear_deadline(&self) {
        *self.lock_deadline() = None;
    }

    pub(crate) fn set_status(&self, status: RunStatus) {
        let current = self.status();
        if matches!(
            current,
            RunStatus::Completed | RunStatus::Cancelled | RunStatus::Failed | RunStatus::Exhausted
        ) && current != status
        {
            return;
        }
        self.0.status.store(status.code(), Ordering::Release);
        self.notify();
    }

    pub(crate) fn cancellation_reason(&self) -> Option<&'static str> {
        if self.status() == RunStatus::Cancelled {
            return Some("run cancelled by host");
        }
        if self
            .lock_deadline()
            .is_some_and(|deadline| Instant::now() >= deadline)
        {
            self.set_status(RunStatus::Cancelled);
            return Some("run deadline exceeded");
        }
        None
    }

    pub(crate) async fn cancelled(&self) -> &'static str {
        let mut changes = self.0.changed.subscribe();
        loop {
            if let Some(reason) = self.cancellation_reason() {
                return reason;
            }
            let deadline = *self.lock_deadline();
            match deadline {
                Some(deadline) => {
                    tokio::select! {
                        _ = changes.changed() => {}
                        _ = tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)) => {}
                    }
                }
                None => {
                    let _ = changes.changed().await;
                }
            }
        }
    }

    pub(crate) async fn wait_if_paused(&self) {
        let mut changes = self.0.changed.subscribe();
        while self.status() == RunStatus::Paused {
            let _ = changes.changed().await;
        }
    }

    pub(crate) fn drain_steer(&self) -> Vec<Message> {
        self.lock_queues().steer.drain(..).collect()
    }

    pub(crate) fn pop_follow_up(&self) -> Option<Message> {
        self.lock_queues().follow_up.pop_front()
    }

    fn notify(&self) {
        self.0
            .changed
            .send_modify(|version| *version = version.wrapping_add(1));
    }

    fn lock_queues(&self) -> std::sync::MutexGuard<'_, Queues> {
        self.0
            .queues
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }

    fn lock_deadline(&self) -> std::sync::MutexGuard<'_, Option<Instant>> {
        self.0
            .deadline
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn queues_are_distinct_and_cancel_wakes_waiters() {
        let control = RunControlHandle::new();
        control.steer("steer");
        control.follow_up("follow");
        control.next_turn("next");
        assert_eq!(control.drain_steer().len(), 1);
        assert!(control.pop_follow_up().is_some());
        assert_eq!(control.drain_next_turn().len(), 1);

        let waiter = {
            let control = control.clone();
            tokio::spawn(async move { control.cancelled().await })
        };
        control.cancel();
        assert_eq!(waiter.await.expect("waiter"), "run cancelled by host");
        assert_eq!(control.status(), RunStatus::Cancelled);
    }

    #[tokio::test]
    async fn pause_and_resume_are_boundary_state() {
        let control = RunControlHandle::new();
        control.set_status(RunStatus::Running);
        control.pause();
        assert_eq!(control.status(), RunStatus::Paused);
        control.resume();
        control.wait_if_paused().await;
        assert_eq!(control.status(), RunStatus::Running);
    }

    #[tokio::test]
    async fn cancellation_before_start_prevents_model_io() {
        use crate::{
            agent::AgentBuilder, completion::PromptError, test_utils::MockCompletionModel,
        };

        let model = MockCompletionModel::text("unused");
        let recorded = model.clone();
        let runner = AgentBuilder::new(model).build().runner("start");
        let control = runner.control_handle();
        control.cancel();
        let error = runner.run().await.expect_err("cancelled run");
        assert!(matches!(error, PromptError::PromptCancelled { .. }));
        assert_eq!(recorded.request_count(), 0);
        assert_eq!(control.status(), RunStatus::Cancelled);
    }

    #[tokio::test]
    async fn steer_and_follow_up_use_separate_safe_boundaries() {
        use crate::{
            agent::AgentBuilder,
            completion::Message,
            test_utils::{MockCompletionModel, MockTurn},
        };

        let model = MockCompletionModel::new([MockTurn::text("first"), MockTurn::text("second")]);
        let recorded = model.clone();
        let runner = AgentBuilder::new(model)
            .build()
            .runner("initial")
            .max_turns(2);
        let control = runner.control_handle();
        control.steer("steered");
        control.follow_up("after settle");
        let response = runner.run().await.expect("controlled run");
        assert_eq!(response.output, "second");

        let requests = recorded.requests();
        assert_eq!(requests.len(), 2);
        let first = requests.first().expect("first request");
        assert_eq!(first.chat_history.last(), Message::user("steered"));
        let second = requests.get(1).expect("follow-up request");
        assert_eq!(second.chat_history.last(), Message::user("after settle"));
        assert_eq!(control.status(), RunStatus::Completed);
    }
}

/// Context automatically attached to every tool call in a run.
#[derive(Debug, Clone)]
pub struct RunContext {
    control: RunControlHandle,
    conversation_id: Option<String>,
}

impl RunContext {
    pub(crate) fn new(control: RunControlHandle, conversation_id: Option<String>) -> Self {
        Self {
            control,
            conversation_id,
        }
    }

    /// Stable run identity.
    pub fn run_id(&self) -> &RunId {
        self.control.run_id()
    }

    /// Durable host conversation identity, when configured.
    pub fn conversation_id(&self) -> Option<&str> {
        self.conversation_id.as_deref()
    }

    /// Cloneable control handle carrying cancellation and deadline state.
    pub fn control(&self) -> &RunControlHandle {
        &self.control
    }
}

/// Correlation metadata automatically attached to one tool invocation.
#[derive(Debug, Clone)]
pub struct ToolCallContext {
    /// Run-scoped context inherited by this call.
    pub run: RunContext,
    /// Rig-generated call identity.
    pub internal_call_id: String,
    /// Provider-generated call identity, when available.
    pub provider_call_id: Option<String>,
    /// Parent call identity for nested dispatch, when applicable.
    pub parent_internal_call_id: Option<String>,
    /// Zero for top-level calls, increasing for nested dispatch.
    pub depth: usize,
}
