//! Hooks for observing and steering an agent run.
//!
//! A hook is a single [`AgentHook::on_event`] method that the agent loop calls
//! at every observable point of a run — before each model call, on each model
//! response, around every tool call, on streamed deltas, and when the model
//! emits an invalid tool call. Each call receives a [`StepEvent`] describing
//! what is happening and returns a [`Flow`] that lets the hook observe, skip a
//! tool, terminate the run early, or (for invalid tool calls) retry/repair/skip
//! recovery.
//!
//! Unlike the old multi-method hook trait, a hook implements one method and
//! matches on the event it cares about — every other event falls through to the
//! default [`Flow::Continue`]. Hooks compose: a [`HookStack`] runs several hooks
//! in registration order and the first non-[`Flow::Continue`] result wins (the
//! later hooks are not consulted for that event).
//!
//! Hooks are a *driver* concern: they are async, side-effecting and generic over
//! the model, so they live in the [`AgentRunner`](crate::agent::AgentRunner)
//! layer rather than inside the sans-IO, serializable
//! [`AgentRun`](crate::agent::run::AgentRun) state machine.
//!
//! # Migrating from `PromptHook`
//!
//! The previous eight-method `PromptHook<M>` trait is replaced by the single
//! [`AgentHook::on_event`] method. Each old method becomes one match arm on a
//! [`StepEvent`] variant, and the value it used to return becomes the [`Flow`]
//! you return from that arm (every event you don't care about falls through to
//! [`Flow::Continue`]). Attach one or more hooks with `add_hook` — they run in
//! registration order and the first non-[`Flow::Continue`] result wins.
//!
//! | Old `PromptHook` method | [`StepEvent`] variant | [`Flow`] to return |
//! |---|---|---|
//! | `on_completion_call` | [`CompletionCall`](StepEvent::CompletionCall) `{ prompt, history, turn }` | [`cont`](Flow::cont) / [`terminate`](Flow::terminate) |
//! | `on_completion_response` | [`CompletionResponse`](StepEvent::CompletionResponse) `{ prompt, response }` | [`cont`](Flow::cont) / [`terminate`](Flow::terminate) |
//! | `on_invalid_tool_call` | [`InvalidToolCall`](StepEvent::InvalidToolCall)`(ctx)` | [`fail`](Flow::fail) (default) / [`retry`](Flow::retry) / [`repair`](Flow::repair) / [`skip`](Flow::skip) / [`terminate`](Flow::terminate) |
//! | `on_tool_call` | [`ToolCall`](StepEvent::ToolCall) `{ tool_name, tool_call_id, internal_call_id, args }` | [`cont`](Flow::cont) / [`skip`](Flow::skip) / [`terminate`](Flow::terminate) |
//! | `on_tool_result` | [`ToolResult`](StepEvent::ToolResult) `{ tool_name, .., result }` | [`cont`](Flow::cont) / [`terminate`](Flow::terminate) |
//! | `on_text_delta` | [`TextDelta`](StepEvent::TextDelta) `{ delta, aggregated }` | [`cont`](Flow::cont) / [`terminate`](Flow::terminate) |
//! | `on_tool_call_delta` | [`ToolCallDelta`](StepEvent::ToolCallDelta) `{ tool_call_id, internal_call_id, tool_name, delta }` | [`cont`](Flow::cont) / [`terminate`](Flow::terminate) |
//! | `on_stream_completion_response_finish` | [`StreamResponseFinish`](StepEvent::StreamResponseFinish) `{ prompt, response }` | [`cont`](Flow::cont) / [`terminate`](Flow::terminate) |
//!
//! Behavioral notes:
//!
//! - The invalid-tool-call default is still fail-fast: returning
//!   [`Flow::Continue`] for [`StepEvent::InvalidToolCall`] is treated as
//!   [`Flow::fail`], matching the old trait's default `on_invalid_tool_call`.
//! - A hook opts out of an event by returning [`Flow::cont`] from that arm,
//!   instead of leaving a trait method unimplemented.
//! - For per-delta hooks, override [`AgentHook::observes`] to skip the
//!   high-frequency [`TextDelta`](StepEvent::TextDelta) /
//!   [`ToolCallDelta`](StepEvent::ToolCallDelta) events you don't consume.

use std::sync::Arc;

use crate::{
    completion::CompletionModel,
    message::{Message, ToolChoice},
    wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync},
};

/// Context passed to a hook on a [`StepEvent::InvalidToolCall`] event when the
/// model emits a tool call that Rig would reject before normal tool-call
/// handling or execution.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct InvalidToolCallContext {
    /// Tool name emitted by the model.
    pub tool_name: String,
    /// Provider-supplied tool call ID, when available.
    pub tool_call_id: Option<String>,
    /// Internal Rig call ID, when available.
    pub internal_call_id: Option<String>,
    /// JSON arguments emitted for the tool call, when available.
    pub args: Option<String>,
    /// Executable Rig tools advertised to the provider for this turn.
    pub available_tools: Vec<String>,
    /// Tools allowed by the active [`ToolChoice`] for this turn.
    pub allowed_tools: Vec<String>,
    /// Active tool choice for this turn.
    pub tool_choice: Option<ToolChoice>,
    /// Diagnostic chat history including the rejected model output when available.
    pub chat_history: Vec<Message>,
    /// Whether the rejected call came from the streaming path.
    pub is_streaming: bool,
}

/// Recovery action for an invalid tool call, used internally by
/// [`AgentRun`](crate::agent::run::AgentRun). Hooks express recovery via
/// [`Flow`]; the [`AgentRunner`](crate::agent::AgentRunner) translates a `Flow`
/// returned for a [`StepEvent::InvalidToolCall`] into this type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvalidToolCallHookAction {
    /// Preserve Rig's default fail-fast behavior.
    Fail,
    /// Retry the model turn with corrective feedback.
    Retry { feedback: String },
    /// Rewrite only the emitted tool name. The repaired name is revalidated
    /// against registered tools and the current `ToolChoice` before use.
    Repair { tool_name: String },
    /// Treat an invalid structured tool call as skipped by returning synthetic
    /// feedback as its tool result. This does not execute the invalid tool.
    Skip { reason: String },
}

impl InvalidToolCallHookAction {
    /// Preserve Rig's default fail-fast behavior.
    pub fn fail() -> Self {
        Self::Fail
    }

    /// Retry the model turn with corrective feedback.
    pub fn retry(feedback: impl Into<String>) -> Self {
        Self::Retry {
            feedback: feedback.into(),
        }
    }

    /// Repair the emitted tool name.
    pub fn repair(tool_name: impl Into<String>) -> Self {
        Self::Repair {
            tool_name: tool_name.into(),
        }
    }

    /// Skip the invalid call with a synthetic tool result.
    pub fn skip(reason: impl Into<String>) -> Self {
        Self::Skip {
            reason: reason.into(),
        }
    }
}

/// An observable point in an agent run, passed to [`AgentHook::on_event`].
///
/// `StepEvent` borrows everything it carries (it is `Copy`), so a hook may
/// inspect the event without taking ownership and a [`HookStack`] can forward
/// the same event to each hook in turn.
///
/// The streaming-only variants ([`TextDelta`](StepEvent::TextDelta),
/// [`ToolCallDelta`](StepEvent::ToolCallDelta) and
/// [`StreamResponseFinish`](StepEvent::StreamResponseFinish)) are emitted only
/// by [`AgentRunner::stream`](crate::agent::AgentRunner::stream).
#[non_exhaustive]
pub enum StepEvent<'a, M: CompletionModel> {
    /// Before a completion request is sent to the model. Honors
    /// [`Flow::Continue`] and [`Flow::Terminate`].
    CompletionCall {
        /// The prompt message for this turn.
        prompt: &'a Message,
        /// The chat history preceding `prompt`.
        history: &'a [Message],
        /// One-based index of this model call within the run.
        turn: usize,
    },
    /// After a non-streaming completion response is received. Suppressed for
    /// turns recovered by invalid tool-call repair, skip, or retry. Honors
    /// [`Flow::Continue`] and [`Flow::Terminate`].
    CompletionResponse {
        /// The prompt message for this turn.
        prompt: &'a Message,
        /// The model's completion response.
        response: &'a crate::completion::CompletionResponse<M::Response>,
    },
    /// The model emitted a tool call that is unknown or disallowed for this
    /// turn. Honors [`Flow::Fail`] (the default), [`Flow::Retry`],
    /// [`Flow::Repair`], [`Flow::Skip`] and [`Flow::Terminate`];
    /// [`Flow::Continue`] is treated as [`Flow::Fail`].
    InvalidToolCall(&'a InvalidToolCallContext),
    /// Before a tool is executed. Honors [`Flow::Continue`], [`Flow::Skip`]
    /// (return `reason` as the tool result without executing) and
    /// [`Flow::Terminate`].
    ToolCall {
        /// Name of the tool about to be called.
        tool_name: &'a str,
        /// Provider-supplied tool call ID, when available.
        tool_call_id: Option<&'a str>,
        /// Internal Rig call ID correlating this call's events.
        internal_call_id: &'a str,
        /// JSON arguments for the call.
        args: &'a str,
    },
    /// After a tool has been executed and produced a result. Honors
    /// [`Flow::Continue`] and [`Flow::Terminate`].
    ToolResult {
        /// Name of the tool that was called.
        tool_name: &'a str,
        /// Provider-supplied tool call ID, when available.
        tool_call_id: Option<&'a str>,
        /// Internal Rig call ID correlating this call's events.
        internal_call_id: &'a str,
        /// JSON arguments for the call.
        args: &'a str,
        /// The tool result, as returned to the model.
        result: &'a str,
    },
    /// Streaming only: a text delta was received. `aggregated` is the full text
    /// accumulated for the turn so far. Honors [`Flow::Continue`] and
    /// [`Flow::Terminate`].
    TextDelta {
        /// The newly received text fragment.
        delta: &'a str,
        /// All text accumulated for the turn so far.
        aggregated: &'a str,
    },
    /// Streaming only: a tool-call delta was received. `tool_name` is `Some` on
    /// the first delta for a tool call and `None` on subsequent deltas. Honors
    /// [`Flow::Continue`] and [`Flow::Terminate`].
    ToolCallDelta {
        /// Provider-supplied tool call ID.
        tool_call_id: &'a str,
        /// Internal Rig call ID correlating this call's events.
        internal_call_id: &'a str,
        /// Tool name, present on the first delta only.
        tool_name: Option<&'a str>,
        /// The newly received argument fragment.
        delta: &'a str,
    },
    /// Streaming only: the provider finished streaming a completion response.
    /// This is the streaming counterpart of [`CompletionResponse`](Self::CompletionResponse)
    /// and, like it, is suppressed for turns recovered by invalid tool-call
    /// repair, skip, or retry. Note one medium-specific difference from
    /// `CompletionResponse`: it fires only on turns that streamed assistant
    /// **text** — a turn that emits only a tool call (or only reasoning) does
    /// not fire it. Honors [`Flow::Continue`] and [`Flow::Terminate`].
    StreamResponseFinish {
        /// The prompt message for this turn.
        prompt: &'a Message,
        /// The provider's final streaming response.
        response: &'a M::StreamingResponse,
    },
}

// `StepEvent` only holds shared references and `Copy` scalars, so it is `Copy`.
// These are hand-written to avoid `derive` adding a spurious `M: Clone`/`M: Copy`
// bound (the generic parameter never appears by value).
impl<M: CompletionModel> Clone for StepEvent<'_, M> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<M: CompletionModel> Copy for StepEvent<'_, M> {}

/// The discriminant of a [`StepEvent`].
///
/// Passed to [`AgentHook::observes`] so a hook can declare which events it cares
/// about without the runner building the (sometimes expensive) event payload —
/// in particular the high-frequency streaming [`TextDelta`](StepEventKind::TextDelta)
/// and [`ToolCallDelta`](StepEventKind::ToolCallDelta) events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum StepEventKind {
    /// [`StepEvent::CompletionCall`].
    CompletionCall,
    /// [`StepEvent::CompletionResponse`].
    CompletionResponse,
    /// [`StepEvent::InvalidToolCall`].
    InvalidToolCall,
    /// [`StepEvent::ToolCall`].
    ToolCall,
    /// [`StepEvent::ToolResult`].
    ToolResult,
    /// [`StepEvent::TextDelta`].
    TextDelta,
    /// [`StepEvent::ToolCallDelta`].
    ToolCallDelta,
    /// [`StepEvent::StreamResponseFinish`].
    StreamResponseFinish,
}

impl<M: CompletionModel> StepEvent<'_, M> {
    /// The [`StepEventKind`] discriminant of this event.
    pub fn kind(&self) -> StepEventKind {
        match self {
            StepEvent::CompletionCall { .. } => StepEventKind::CompletionCall,
            StepEvent::CompletionResponse { .. } => StepEventKind::CompletionResponse,
            StepEvent::InvalidToolCall(_) => StepEventKind::InvalidToolCall,
            StepEvent::ToolCall { .. } => StepEventKind::ToolCall,
            StepEvent::ToolResult { .. } => StepEventKind::ToolResult,
            StepEvent::TextDelta { .. } => StepEventKind::TextDelta,
            StepEvent::ToolCallDelta { .. } => StepEventKind::ToolCallDelta,
            StepEvent::StreamResponseFinish { .. } => StepEventKind::StreamResponseFinish,
        }
    }
}

/// Control-flow result returned by [`AgentHook::on_event`].
///
/// Each [`StepEvent`] honors a specific subset of variants (documented on each
/// event). The runner is **fail-closed**: an action an event cannot honor never
/// silently proceeds — it terminates the run with a diagnostic error. In
/// particular, a blocking action such as [`Flow::Fail`] returned for a
/// [`StepEvent::ToolCall`] stops the run rather than letting the tool execute.
/// Returning [`Flow::Continue`] is always the way to "do nothing".
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Flow {
    /// Proceed normally.
    Continue,
    /// Terminate the agent run early, surfacing `reason`.
    Terminate {
        /// Why the run is being terminated.
        reason: String,
    },
    /// Skip the action: for [`StepEvent::ToolCall`], return `reason` as the tool
    /// result without executing the tool; for [`StepEvent::InvalidToolCall`],
    /// record `reason` as a synthetic result for the invalid call.
    Skip {
        /// The message returned to the model in place of the tool result.
        reason: String,
    },
    /// [`StepEvent::InvalidToolCall`] only: fail the run fast (the default for
    /// invalid tool calls).
    Fail,
    /// [`StepEvent::InvalidToolCall`] only: retry the model turn with corrective
    /// feedback.
    Retry {
        /// Feedback appended to the conversation before re-prompting.
        feedback: String,
    },
    /// [`StepEvent::InvalidToolCall`] only: rewrite the emitted tool name, which
    /// is then revalidated against the allowed tools.
    Repair {
        /// The corrected tool name.
        tool_name: String,
    },
}

impl Flow {
    /// Continue the agent loop as normal.
    pub fn cont() -> Self {
        Self::Continue
    }

    /// Terminate the agent run early with a reason.
    pub fn terminate(reason: impl Into<String>) -> Self {
        Self::Terminate {
            reason: reason.into(),
        }
    }

    /// Skip the current tool call (or invalid call) with the provided reason.
    pub fn skip(reason: impl Into<String>) -> Self {
        Self::Skip {
            reason: reason.into(),
        }
    }

    /// Fail fast on an invalid tool call (the default).
    pub fn fail() -> Self {
        Self::Fail
    }

    /// Retry the model turn with corrective feedback (invalid tool calls only).
    pub fn retry(feedback: impl Into<String>) -> Self {
        Self::Retry {
            feedback: feedback.into(),
        }
    }

    /// Repair the emitted tool name (invalid tool calls only).
    pub fn repair(tool_name: impl Into<String>) -> Self {
        Self::Repair {
            tool_name: tool_name.into(),
        }
    }
}

// ── Typed step controls ───────────────────────────────────────────────────
//
// Every observable point of a run honours only a subset of [`Flow`] (documented
// on each [`StepEvent`] variant). Rather than re-deriving "can this event honour
// that action?" ad hoc at each driver call site — and falling back to a
// stringly-typed `Terminate` when it can't — the admissible subset is encoded as
// a distinct `Control` type per step. A `Control` is `Into<Flow>` (the single
// erased runtime value the hook surface speaks) and is produced by an
// *exhaustive*, fallback-free [`Step::parse`] of a hook's `Flow`. The fail-closed
// decision is therefore made in exactly one place — the type — and the per-event
// resolvers in the driver become total folds with no unreachable arm.
//
// `on_event(StepEvent) -> Flow` stays the ergonomic hook surface; these types are
// the internal, typed layer the runner drives through, and are exposed so an
// advanced hook can construct an admissible action without round-tripping the
// untyped `Flow`.

/// Admissible control at an observe-only step — a completion call/response,
/// tool result, or any streamed delta / stream-finish event. Such a step may
/// only let the run continue or stop it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObserveControl {
    /// Proceed normally.
    Continue,
    /// Terminate the run, surfacing `reason`.
    Terminate(String),
}

/// Admissible control at a tool-call step: proceed, skip the tool (returning
/// `reason` to the model in place of executing it), or stop the run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolControl {
    /// Execute the tool as normal.
    Continue,
    /// Skip execution and return `reason` to the model as the tool result.
    Skip(String),
    /// Terminate the run.
    Terminate(String),
}

/// Admissible control at an invalid-tool-call step: the full recovery set.
/// Every [`Flow`] is admissible here (`Continue` collapses to `Fail`, the
/// documented fail-fast default), so this step's parse is total.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoveryControl {
    /// Preserve Rig's fail-fast default.
    Fail,
    /// Retry the model turn with corrective feedback.
    Retry(String),
    /// Rewrite the emitted tool name (revalidated against allowed tools).
    Repair(String),
    /// Record `reason` as a synthetic result for the invalid call.
    Skip(String),
    /// Terminate the run.
    Terminate(String),
}

impl From<ObserveControl> for Flow {
    fn from(control: ObserveControl) -> Self {
        match control {
            ObserveControl::Continue => Flow::Continue,
            ObserveControl::Terminate(reason) => Flow::Terminate { reason },
        }
    }
}

impl From<ToolControl> for Flow {
    fn from(control: ToolControl) -> Self {
        match control {
            ToolControl::Continue => Flow::Continue,
            ToolControl::Skip(reason) => Flow::Skip { reason },
            ToolControl::Terminate(reason) => Flow::Terminate { reason },
        }
    }
}

impl From<RecoveryControl> for Flow {
    fn from(control: RecoveryControl) -> Self {
        match control {
            RecoveryControl::Fail => Flow::Fail,
            RecoveryControl::Retry(feedback) => Flow::Retry { feedback },
            RecoveryControl::Repair(tool_name) => Flow::Repair { tool_name },
            RecoveryControl::Skip(reason) => Flow::Skip { reason },
            RecoveryControl::Terminate(reason) => Flow::Terminate { reason },
        }
    }
}

impl From<RecoveryControl> for InvalidToolCallHookAction {
    fn from(control: RecoveryControl) -> Self {
        match control {
            RecoveryControl::Fail => InvalidToolCallHookAction::Fail,
            RecoveryControl::Retry(feedback) => InvalidToolCallHookAction::Retry { feedback },
            RecoveryControl::Repair(tool_name) => InvalidToolCallHookAction::Repair { tool_name },
            RecoveryControl::Skip(reason) => InvalidToolCallHookAction::Skip { reason },
            // `Terminate` is not a recovery action; the driver intercepts it
            // before this conversion (it stops the run rather than resolving the
            // invalid call). Mapped to the fail-fast default for totality.
            RecoveryControl::Terminate(_) => InvalidToolCallHookAction::Fail,
        }
    }
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::ObserveStep {}
    impl Sealed for super::ToolCallStep {}
    impl Sealed for super::InvalidToolCallStep {}
}

/// A node of the run's execution graph, identifying which [`Flow`] actions are
/// admissible there. The set of steps is closed (`sealed`); each carries its
/// typed [`Control`](Step::Control) and an exhaustive [`parse`](Step::parse) of a
/// hook's `Flow` into it. `Err(flow)` marks an inadmissible action the driver
/// fails closed on.
pub trait Step: sealed::Sealed {
    /// The typed control admissible at this step.
    type Control: Into<Flow>;
    /// Human-readable name of the step, for fail-closed diagnostics.
    const LABEL: &'static str;
    /// The `Flow` actions this step honours, for fail-closed diagnostics.
    const HONORS: &'static str;
    /// Total parse of a hook's [`Flow`] into this step's admissible control.
    fn parse(flow: Flow) -> Result<Self::Control, Flow>;
}

/// The observe-only step (see [`ObserveControl`]).
pub struct ObserveStep;
/// The tool-call step (see [`ToolControl`]).
pub struct ToolCallStep;
/// The invalid-tool-call step (see [`RecoveryControl`]).
pub struct InvalidToolCallStep;

impl Step for ObserveStep {
    type Control = ObserveControl;
    const LABEL: &'static str = "an observe-only event";
    const HONORS: &'static str = "Continue/Terminate";
    fn parse(flow: Flow) -> Result<ObserveControl, Flow> {
        match flow {
            Flow::Continue => Ok(ObserveControl::Continue),
            Flow::Terminate { reason } => Ok(ObserveControl::Terminate(reason)),
            other => Err(other),
        }
    }
}

impl Step for ToolCallStep {
    type Control = ToolControl;
    const LABEL: &'static str = "a tool-call event";
    const HONORS: &'static str = "Continue/Skip/Terminate";
    fn parse(flow: Flow) -> Result<ToolControl, Flow> {
        match flow {
            Flow::Continue => Ok(ToolControl::Continue),
            Flow::Skip { reason } => Ok(ToolControl::Skip(reason)),
            Flow::Terminate { reason } => Ok(ToolControl::Terminate(reason)),
            other => Err(other),
        }
    }
}

impl Step for InvalidToolCallStep {
    type Control = RecoveryControl;
    const LABEL: &'static str = "an invalid-tool-call event";
    const HONORS: &'static str = "Fail/Retry/Repair/Skip/Terminate";
    fn parse(flow: Flow) -> Result<RecoveryControl, Flow> {
        // Total: every `Flow` is admissible at an invalid-tool-call step, so this
        // never returns `Err`. `Continue` collapses to the fail-fast default.
        Ok(match flow {
            Flow::Continue | Flow::Fail => RecoveryControl::Fail,
            Flow::Retry { feedback } => RecoveryControl::Retry(feedback),
            Flow::Repair { tool_name } => RecoveryControl::Repair(tool_name),
            Flow::Skip { reason } => RecoveryControl::Skip(reason),
            Flow::Terminate { reason } => RecoveryControl::Terminate(reason),
        })
    }
}

/// A per-run hook that observes and steers an agent run.
///
/// Implement [`on_event`](AgentHook::on_event) and match on the [`StepEvent`]
/// variants you care about; every other event falls through to the default
/// [`Flow::Continue`]. Hooks must be cheap to share (`Clone` is not required —
/// hooks are held behind an `Arc` once registered).
///
/// # Example
/// ```rust,ignore
/// use rig_core::agent::{AgentHook, Flow, StepEvent};
/// use rig_core::completion::CompletionModel;
///
/// #[derive(Clone)]
/// struct Logger;
///
/// impl<M: CompletionModel> AgentHook<M> for Logger {
///     async fn on_event(&self, event: StepEvent<'_, M>) -> Flow {
///         if let StepEvent::ToolCall { tool_name, args, .. } = event {
///             println!("calling {tool_name}({args})");
///         }
///         Flow::cont()
///     }
/// }
/// ```
pub trait AgentHook<M>: WasmCompatSend + WasmCompatSync
where
    M: CompletionModel,
{
    /// Called at every observable point of the agent run (subject to
    /// [`observes`](Self::observes)). The default implementation observes
    /// nothing and returns [`Flow::Continue`].
    fn on_event(&self, event: StepEvent<'_, M>) -> impl Future<Output = Flow> + WasmCompatSend {
        let _ = event;
        async { Flow::Continue }
    }

    /// Whether this hook observes events of the given [`StepEventKind`].
    ///
    /// This is a **performance hint for the high-frequency streaming
    /// [`TextDelta`](StepEventKind::TextDelta) /
    /// [`ToolCallDelta`](StepEventKind::ToolCallDelta) events**, which otherwise
    /// cost one boxed future per delta. The runner skips building and
    /// dispatching a delta event only when *no* hook in the stack observes it
    /// (interest is OR-combined across the stack), so a hook may still be
    /// invoked for a delta a sibling observes — `on_event` must therefore stay
    /// total (return [`Flow::Continue`] for events it ignores) rather than
    /// assume it is only called for observed kinds.
    ///
    /// Control flow is **never** changed by `observes`: the shared, steering
    /// events ([`ToolCall`](StepEventKind::ToolCall),
    /// [`InvalidToolCall`](StepEventKind::InvalidToolCall), …) fire identically
    /// regardless of this method, so `run()` and `stream()` stay in lock-step.
    /// The default observes everything.
    fn observes(&self, kind: StepEventKind) -> bool {
        let _ = kind;
        true
    }
}

/// The no-op hook: observes nothing, never alters control flow.
impl<M> AgentHook<M> for () where M: CompletionModel {}

/// Object-safe shim over [`AgentHook`] so a [`HookStack`] can hold a
/// heterogeneous list of hooks behind `Arc`.
trait DynAgentHook<M>: WasmCompatSend + WasmCompatSync
where
    M: CompletionModel,
{
    fn on_event_boxed<'a>(&'a self, event: StepEvent<'a, M>) -> WasmBoxedFuture<'a, Flow>
    where
        M: 'a;

    fn observes_dyn(&self, kind: StepEventKind) -> bool;
}

impl<M, H> DynAgentHook<M> for H
where
    M: CompletionModel,
    H: AgentHook<M>,
{
    fn on_event_boxed<'a>(&'a self, event: StepEvent<'a, M>) -> WasmBoxedFuture<'a, Flow>
    where
        M: 'a,
    {
        Box::pin(self.on_event(event))
    }

    fn observes_dyn(&self, kind: StepEventKind) -> bool {
        self.observes(kind)
    }
}

/// An ordered list of hooks run as one hook.
///
/// Each hook is consulted in registration order; the first hook that returns a
/// non-[`Flow::Continue`] result short-circuits the rest for that event. Because
/// the runner is fail-closed (a non-`Continue` action always takes effect or
/// terminates the run — it is never silently ignored), short-circuiting is
/// always meaningful: a later hook is only skipped for an event an earlier hook
/// actually steered. An empty stack is the no-op hook and
/// [`observes`](HookStack::observes) nothing, so the runner skips event dispatch
/// for it entirely.
///
/// This is the default hook type carried by an
/// [`Agent`](crate::agent::Agent) and an
/// [`AgentRunner`](crate::agent::AgentRunner); build one with
/// [`add_hook`](crate::agent::AgentRunner::add_hook).
pub struct HookStack<M>
where
    M: CompletionModel,
{
    hooks: Vec<Arc<dyn DynAgentHook<M>>>,
}

// Hand-written so the impls do not require `M: Clone`/`M: Default`: `M` only
// appears inside `Arc<dyn DynAgentHook<M>>`, never by value.
impl<M> Clone for HookStack<M>
where
    M: CompletionModel,
{
    fn clone(&self) -> Self {
        Self {
            hooks: self.hooks.clone(),
        }
    }
}

impl<M> Default for HookStack<M>
where
    M: CompletionModel,
{
    fn default() -> Self {
        Self { hooks: Vec::new() }
    }
}

impl<M> std::fmt::Debug for HookStack<M>
where
    M: CompletionModel,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HookStack")
            .field("len", &self.hooks.len())
            .finish()
    }
}

impl<M> HookStack<M>
where
    M: CompletionModel,
{
    /// An empty stack (the no-op hook).
    pub fn new() -> Self {
        Self::default()
    }

    /// A stack containing a single hook.
    pub fn with<H>(hook: H) -> Self
    where
        H: AgentHook<M> + 'static,
    {
        let mut stack = Self::new();
        stack.push(hook);
        stack
    }

    /// Append a hook to the end of the stack.
    pub fn push<H>(&mut self, hook: H)
    where
        H: AgentHook<M> + 'static,
    {
        self.hooks.push(Arc::new(hook));
    }

    /// Whether the stack contains no hooks.
    pub fn is_empty(&self) -> bool {
        self.hooks.is_empty()
    }

    /// Number of hooks in the stack.
    pub fn len(&self) -> usize {
        self.hooks.len()
    }
}

impl<M> AgentHook<M> for HookStack<M>
where
    M: CompletionModel,
{
    async fn on_event(&self, event: StepEvent<'_, M>) -> Flow {
        for hook in &self.hooks {
            match hook.on_event_boxed(event).await {
                Flow::Continue => {}
                other => return other,
            }
        }
        Flow::Continue
    }

    /// The stack observes an event kind if any of its hooks does (so an empty
    /// stack observes nothing).
    fn observes(&self, kind: StepEventKind) -> bool {
        self.hooks.iter().any(|hook| hook.observes_dyn(kind))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::{AgentHook, Flow, HookStack, StepEvent, StepEventKind};
    use crate::test_utils::MockCompletionModel;

    type M = MockCompletionModel;

    /// Pushes its label when invoked and returns `Continue` or `Terminate`.
    struct Recorder {
        label: u32,
        log: Arc<Mutex<Vec<u32>>>,
        stop: bool,
    }

    impl AgentHook<M> for Recorder {
        async fn on_event(&self, _event: StepEvent<'_, M>) -> Flow {
            self.log.lock().expect("log").push(self.label);
            if self.stop {
                Flow::terminate("stop")
            } else {
                Flow::cont()
            }
        }
    }

    /// Observes exactly one event kind (used to probe stack-level `observes`).
    struct ObservesOnly(StepEventKind);

    impl AgentHook<M> for ObservesOnly {
        async fn on_event(&self, _event: StepEvent<'_, M>) -> Flow {
            Flow::cont()
        }

        fn observes(&self, kind: StepEventKind) -> bool {
            kind == self.0
        }
    }

    /// A cheap, M-agnostic event to dispatch (no model response required).
    fn tool_call_event() -> StepEvent<'static, M> {
        StepEvent::ToolCall {
            tool_name: "add",
            tool_call_id: Some("tc1"),
            internal_call_id: "ic1",
            args: "{}",
        }
    }

    #[tokio::test]
    async fn runs_hooks_in_registration_order_and_consults_all_on_continue() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut stack = HookStack::<M>::with(Recorder {
            label: 1,
            log: log.clone(),
            stop: false,
        });
        stack.push(Recorder {
            label: 2,
            log: log.clone(),
            stop: false,
        });

        let flow = stack.on_event(tool_call_event()).await;

        assert!(matches!(flow, Flow::Continue));
        assert_eq!(*log.lock().expect("log"), vec![1, 2]);
    }

    #[tokio::test]
    async fn first_non_continue_short_circuits_the_rest() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut stack = HookStack::<M>::with(Recorder {
            label: 1,
            log: log.clone(),
            stop: true,
        });
        stack.push(Recorder {
            label: 2,
            log: log.clone(),
            stop: false,
        });

        let flow = stack.on_event(tool_call_event()).await;

        assert!(matches!(flow, Flow::Terminate { .. }));
        assert_eq!(
            *log.lock().expect("log"),
            vec![1],
            "a later hook must not run after an earlier hook returns non-Continue"
        );
    }

    #[test]
    fn stack_observes_is_the_or_of_its_members() {
        let mut stack = HookStack::<M>::with(ObservesOnly(StepEventKind::ToolCall));
        stack.push(ObservesOnly(StepEventKind::ToolResult));

        assert!(stack.observes(StepEventKind::ToolCall));
        assert!(stack.observes(StepEventKind::ToolResult));
        assert!(
            !stack.observes(StepEventKind::TextDelta),
            "no member observes TextDelta, so the stack must not either"
        );
    }

    #[tokio::test]
    async fn empty_stack_continues_and_observes_nothing() {
        let stack = HookStack::<M>::new();

        assert!(stack.is_empty());
        assert!(!stack.observes(StepEventKind::ToolCall));
        assert!(!stack.observes(StepEventKind::TextDelta));
        assert!(matches!(
            stack.on_event(tool_call_event()).await,
            Flow::Continue
        ));
    }

    // ── Typed step controls ───────────────────────────────────────────────

    use super::{
        InvalidToolCallStep, ObserveControl, ObserveStep, RecoveryControl, Step, ToolCallStep,
        ToolControl,
    };

    #[test]
    fn observe_step_admits_continue_terminate_and_rejects_the_rest() {
        assert_eq!(
            ObserveStep::parse(Flow::cont()),
            Ok(ObserveControl::Continue)
        );
        assert_eq!(
            ObserveStep::parse(Flow::terminate("stop")),
            Ok(ObserveControl::Terminate("stop".into()))
        );
        // A steering action an observe-only step cannot honor is returned as
        // `Err(flow)` for the driver to fail closed on — never silently dropped.
        assert_eq!(
            ObserveStep::parse(Flow::skip("nope")),
            Err(Flow::skip("nope"))
        );
        assert_eq!(ObserveStep::parse(Flow::fail()), Err(Flow::fail()));
    }

    #[test]
    fn tool_call_step_admits_continue_skip_terminate_only() {
        assert_eq!(ToolCallStep::parse(Flow::cont()), Ok(ToolControl::Continue));
        assert_eq!(
            ToolCallStep::parse(Flow::skip("policy")),
            Ok(ToolControl::Skip("policy".into()))
        );
        assert_eq!(
            ToolCallStep::parse(Flow::terminate("halt")),
            Ok(ToolControl::Terminate("halt".into()))
        );
        // `Fail`/`Retry`/`Repair` are inadmissible at a tool call → fail-closed.
        assert_eq!(ToolCallStep::parse(Flow::fail()), Err(Flow::fail()));
        assert_eq!(
            ToolCallStep::parse(Flow::repair("add")),
            Err(Flow::repair("add"))
        );
    }

    #[test]
    fn invalid_tool_call_step_is_total_and_continue_collapses_to_fail() {
        // Every `Flow` is admissible here; `Continue` and `Fail` both fail fast.
        assert_eq!(
            InvalidToolCallStep::parse(Flow::cont()),
            Ok(RecoveryControl::Fail)
        );
        assert_eq!(
            InvalidToolCallStep::parse(Flow::fail()),
            Ok(RecoveryControl::Fail)
        );
        assert_eq!(
            InvalidToolCallStep::parse(Flow::retry("fix it")),
            Ok(RecoveryControl::Retry("fix it".into()))
        );
        assert_eq!(
            InvalidToolCallStep::parse(Flow::repair("add")),
            Ok(RecoveryControl::Repair("add".into()))
        );
        assert_eq!(
            InvalidToolCallStep::parse(Flow::skip("synthetic")),
            Ok(RecoveryControl::Skip("synthetic".into()))
        );
        assert_eq!(
            InvalidToolCallStep::parse(Flow::terminate("done")),
            Ok(RecoveryControl::Terminate("done".into()))
        );
    }

    #[test]
    fn controls_round_trip_into_flow() {
        // Each typed control erases back to exactly the `Flow` it came from, so
        // the typed layer adds no observable behavior to the hook surface.
        assert_eq!(Flow::from(ObserveControl::Continue), Flow::cont());
        assert_eq!(Flow::from(ToolControl::Skip("r".into())), Flow::skip("r"));
        assert_eq!(
            Flow::from(RecoveryControl::Repair("add".into())),
            Flow::repair("add")
        );
    }
}
