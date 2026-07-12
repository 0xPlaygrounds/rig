//! Host-facing control for an in-flight agent run.
//!
//! A [`RunControlHandle`] is bound to one runner invocation. It provides stable
//! correlation, cooperative cancellation, steering, and queues that a host can
//! consume after settlement without storing messages on a shared [`Agent`](super::Agent).

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU8, Ordering},
};

use futures::task::AtomicWaker;

use super::{MessageInjectError, MessageInjector, hook::RunId};
use crate::completion::Message;

/// Observable lifecycle state of an agent run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RunStatus {
    /// The runner has been configured but has not started.
    Pending,
    /// Model/tool work is in progress.
    Running,
    /// The run produced a final response.
    Completed,
    /// The host cancelled the run.
    Cancelled,
    /// The model-call budget was exhausted.
    Exhausted,
    /// The run failed for another reason.
    Failed,
}

impl RunStatus {
    fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::Pending,
            1 => Self::Running,
            2 => Self::Completed,
            3 => Self::Cancelled,
            4 => Self::Exhausted,
            _ => Self::Failed,
        }
    }

    fn as_u8(self) -> u8 {
        match self {
            Self::Pending => 0,
            Self::Running => 1,
            Self::Completed => 2,
            Self::Cancelled => 3,
            Self::Exhausted => 4,
            Self::Failed => 5,
        }
    }

    /// Whether no more work will be started for this run.
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Cancelled | Self::Exhausted | Self::Failed
        )
    }
}

#[derive(Debug)]
pub(crate) struct RunControlState {
    run_id: RunId,
    cancelled: AtomicBool,
    cancel_waker: AtomicWaker,
    status: AtomicU8,
    deadline: Mutex<Option<std::time::SystemTime>>,
    follow_ups: Mutex<Vec<Message>>,
    next_turn: Mutex<Vec<Message>>,
}

/// Cloneable host handle for one agent run.
#[derive(Clone, Debug)]
pub struct RunControlHandle {
    pub(crate) state: Arc<RunControlState>,
    steer: MessageInjector,
}

impl RunControlHandle {
    pub(crate) fn new(steer: MessageInjector) -> Self {
        Self {
            state: Arc::new(RunControlState {
                run_id: RunId::generate(),
                cancelled: AtomicBool::new(false),
                cancel_waker: AtomicWaker::new(),
                status: AtomicU8::new(RunStatus::Pending.as_u8()),
                deadline: Mutex::new(None),
                follow_ups: Mutex::new(Vec::new()),
                next_turn: Mutex::new(Vec::new()),
            }),
            steer,
        }
    }

    /// Stable process-scoped identifier for this run.
    pub fn run_id(&self) -> &RunId {
        &self.state.run_id
    }

    /// Current lifecycle state.
    pub fn status(&self) -> RunStatus {
        RunStatus::from_u8(self.state.status.load(Ordering::Acquire))
    }

    /// Request cooperative cancellation.
    ///
    /// The shared driver stops polling the active model/tool future and does not
    /// start later work. Tools can also read [`RunContext`] from call extensions.
    pub fn cancel(&self) {
        self.state.cancelled.store(true, Ordering::Release);
        self.state.cancel_waker.wake();
    }

    /// Whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.state.cancelled.load(Ordering::Acquire) || self.deadline_exceeded()
    }

    /// Set an absolute cooperative deadline for model, tool, and child work.
    pub fn set_deadline(&self, deadline: std::time::SystemTime) {
        *self
            .state
            .deadline
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(deadline);
        self.state.cancel_waker.wake();
    }

    /// Configured absolute deadline.
    pub fn deadline(&self) -> Option<std::time::SystemTime> {
        *self
            .state
            .deadline
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    /// Whether the configured deadline has passed.
    pub fn deadline_exceeded(&self) -> bool {
        self.deadline()
            .is_some_and(|deadline| std::time::SystemTime::now() >= deadline)
    }

    /// Deliver a user message at the next safe boundary before a model call.
    pub fn steer(&self, message: impl Into<Message>) -> Result<(), MessageInjectError> {
        self.steer.inject(message)
    }

    /// Queue a message for a host-started follow-up after this run settles.
    pub fn follow_up(&self, message: impl Into<Message>) {
        self.state
            .follow_ups
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(message.into());
    }

    /// Queue input for a later turn without interrupting or starting this run.
    pub fn queue_next_turn(&self, message: impl Into<Message>) {
        self.state
            .next_turn
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(message.into());
    }

    /// Drain queued follow-up messages in producer order.
    pub fn take_follow_ups(&self) -> Vec<Message> {
        std::mem::take(
            &mut *self
                .state
                .follow_ups
                .lock()
                .unwrap_or_else(|e| e.into_inner()),
        )
    }

    /// Drain queued next-turn messages in producer order.
    pub fn take_next_turn(&self) -> Vec<Message> {
        std::mem::take(
            &mut *self
                .state
                .next_turn
                .lock()
                .unwrap_or_else(|e| e.into_inner()),
        )
    }

    pub(crate) fn message_injector(&self) -> MessageInjector {
        self.steer.clone()
    }

    pub(crate) fn set_status(&self, status: RunStatus) {
        self.state.status.store(status.as_u8(), Ordering::Release);
    }

    pub(crate) fn status_guard(&self) -> RunStatusGuard {
        RunStatusGuard(self.clone())
    }

    pub(crate) async fn cancelled(&self) {
        let explicit_cancel = std::future::poll_fn(|cx| {
            if self.state.cancelled.load(Ordering::Acquire) {
                return std::task::Poll::Ready(());
            }
            self.state.cancel_waker.register(cx.waker());
            if self.state.cancelled.load(Ordering::Acquire) {
                std::task::Poll::Ready(())
            } else {
                std::task::Poll::Pending
            }
        });

        if let Some(deadline) = self.deadline() {
            let duration = deadline
                .duration_since(std::time::SystemTime::now())
                .unwrap_or_default();
            futures::pin_mut!(explicit_cancel);
            let deadline = tokio::time::sleep(duration);
            futures::pin_mut!(deadline);
            let _ = futures::future::select(explicit_cancel, deadline).await;
        } else {
            explicit_cancel.await;
        }
    }
}

/// Marks an abandoned driver/stream as cancelled when its future is dropped.
pub(crate) struct RunStatusGuard(RunControlHandle);

impl Drop for RunStatusGuard {
    fn drop(&mut self) {
        if !self.0.status().is_terminal() {
            self.0.set_status(RunStatus::Cancelled);
            self.0.cancel();
        }
    }
}

/// Framework-populated, tool-visible context for the current run.
///
/// It is inserted into [`ToolCallExtensions`](crate::tool::ToolCallExtensions)
/// automatically. It remains host-only and is never serialized into model input.
#[derive(Clone, Debug)]
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

    /// Stable process-scoped run identifier.
    pub fn run_id(&self) -> &RunId {
        self.control.run_id()
    }

    /// Durable host conversation identifier, when configured.
    pub fn conversation_id(&self) -> Option<&str> {
        self.conversation_id.as_deref()
    }

    /// Whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.control.is_cancelled()
    }

    /// Configured absolute deadline.
    pub fn deadline(&self) -> Option<std::time::SystemTime> {
        self.control.deadline()
    }

    /// Clone the host control handle.
    pub fn control(&self) -> RunControlHandle {
        self.control.clone()
    }
}
