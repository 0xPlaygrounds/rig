//! Inject messages into an agent run while it is in flight.
//!
//! A long-running agent — a multi-turn tool loop or a background worker —
//! sometimes needs to hear about something that happened *after* it was
//! prompted: a user follow-up, an external event, a cancellation note.
//! [`MessageInjector`] is the handle for that — a cloneable, `Send` mailbox
//! bound to a single run. A message pushed through it is delivered as its own
//! user turn immediately before the run's next model call.
//!
//! ```rust,ignore
//! let mut runner = agent.runner("Investigate and report.").max_turns(10);
//! let injector = runner.message_injector();
//!
//! let run = tokio::spawn(async move { runner.run().await });
//!
//! // ...from a watcher task, when something happens externally:
//! injector.inject("Heads up: the deploy just failed, factor that in.")?;
//!
//! let response = run.await??;
//! ```
//!
//! # Delivery semantics
//!
//! Injection is **best-effort, in-flight delivery**: a message is folded into
//! the conversation immediately before the next model call (any tool calls
//! already in flight on the current turn finish first). If the run *finishes*
//! before that next model call — i.e. there is no further turn to deliver on —
//! the message is not delivered, and a subsequent [`inject`](MessageInjector::inject)
//! returns [`MessageInjectError::RunFinished`]. Injection does not resurrect a
//! completed run; to continue a finished conversation, start a new run with the
//! prior history.
//!
//! The message is delivered as a **separate user turn**, never merged into a
//! preceding tool-result message: that mixed shape is dropped by OpenAI's
//! converter and is non-idiomatic on Anthropic, whereas consecutive user turns
//! are combined by Anthropic and accepted by OpenAI.
//!
//! # Design
//!
//! The mechanism splits cleanly across the two layers: the sans-IO [`AgentRun`]
//! owns *folding* injected messages into the conversation
//! ([`AgentRun::inject_message`]) as a plain, serializable state transition a
//! hand-driver can call; the async [`AgentRunner`](crate::agent::AgentRunner)
//! owns the *transport* — a [`futures::channel::mpsc`] queue drained before each
//! model call. Both the blocking and streaming drivers use the same helpers, so
//! injection behaves identically.
//!
//! The "fold input between steps" point mirrors pydantic-ai's `AgentRun.enqueue`,
//! Vercel AI SDK's `prepareStep`, and LangGraph's `Command(resume=…)` — but needs
//! no checkpointer: delivery is in-process for the lifetime of the run.

use futures::channel::mpsc::{UnboundedReceiver, UnboundedSender, unbounded};

use crate::agent::run::AgentRun;
use crate::completion::Message;

/// A cloneable handle for injecting messages into a single in-flight agent run.
///
/// Obtain one from [`AgentRunner::message_injector`](crate::agent::AgentRunner::message_injector)
/// (or the equivalent on [`PromptRequest`](crate::agent::PromptRequest) /
/// [`StreamingPromptRequest`](crate::agent::prompt_request::streaming::StreamingPromptRequest))
/// *before* the run starts, then move the runner into a task and push from
/// anywhere holding the handle. Cloning yields another producer for the same
/// run, so several tasks can inject concurrently.
///
/// The handle goes inert once the run ends: [`inject`](Self::inject) then returns
/// [`MessageInjectError::RunFinished`].
#[derive(Clone, Debug)]
pub struct MessageInjector {
    tx: UnboundedSender<Message>,
}

impl MessageInjector {
    /// Queue a message for delivery to the run this handle is bound to.
    ///
    /// Delivery is best-effort and in-flight: the message is folded into the
    /// conversation as a user turn before the run's next model call (tool calls
    /// already in flight on the current turn finish first). It accepts anything
    /// that converts into a [`Message`]; a `&str`/`String` becomes a user
    /// message, which is the intended shape.
    ///
    /// # Errors
    /// Returns [`MessageInjectError::RunFinished`] if the run has already
    /// finished — or if the runner was dropped without being started — so the
    /// receiver is gone and the message cannot be delivered. (Injecting *before*
    /// the run starts is fine: the message is folded into the first turn.) Note
    /// that a message queued while the run is still active but which the run
    /// finishes before consuming is dropped (and logged) rather than reported
    /// through this call — see the [module docs](self).
    pub fn inject(&self, message: impl Into<Message>) -> Result<(), MessageInjectError> {
        self.tx
            .unbounded_send(message.into())
            .map_err(|_| MessageInjectError::RunFinished)
    }

    /// Whether the bound run is still accepting injected messages.
    ///
    /// A best-effort hint: the run may finish between this check and the next
    /// [`inject`](Self::inject), so a `true` result does not guarantee delivery.
    pub fn is_active(&self) -> bool {
        !self.tx.is_closed()
    }
}

/// Why a [`MessageInjector::inject`] call could not accept its message.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum MessageInjectError {
    /// The run has already finished (or never started), so its receiver is gone.
    #[error("cannot inject a message: the agent run has already finished")]
    RunFinished,
}

/// Create a paired injector handle and the receiver the driver drains. The
/// driver keeps the receiver for the run's lifetime; dropping it (when the run
/// finishes) is what makes a later [`MessageInjector::inject`] fail.
pub(crate) fn injector_channel() -> (MessageInjector, UnboundedReceiver<Message>) {
    let (tx, rx) = unbounded();
    (MessageInjector { tx }, rx)
}

/// Drain every message currently queued on `rx` into `run` (a no-op when no
/// injector was attached). Called at the top of each drive-loop iteration, so
/// queued messages are folded in by the next [`AgentRun::next_step`].
pub(crate) fn drain_injections(rx: &mut Option<UnboundedReceiver<Message>>, run: &mut AgentRun) {
    let Some(rx) = rx.as_mut() else {
        return;
    };
    // `try_recv` yields `Ok(_)` per ready message and `Err` once the queue is
    // empty (or every sender has dropped); either stops the drain.
    while let Ok(message) = rx.try_recv() {
        run.inject_message(message);
    }
}
