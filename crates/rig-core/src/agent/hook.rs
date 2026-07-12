//! Hooks for observing and steering an agent run.
//!
//! A hook is a single [`AgentHook::on_event`] method that the agent loop calls
//! at every observable point of a run — before each model call, on each model
//! response, around every tool call, on streamed deltas, and when the model
//! emits an invalid tool call. Each call receives a [`HookContext`] (run-scoped
//! identity + a shared scratchpad) and a [`StepEvent`] describing what is
//! happening, and returns a [`Flow`] that lets the hook observe, patch the
//! request, skip a tool, terminate the run early, or (for invalid tool calls)
//! retry/repair/skip recovery.
//!
//! Unlike the old multi-method hook trait, a hook implements one method and
//! matches on the event it cares about — every other event falls through to the
//! default [`Flow::Continue`]. Hooks compose in a [`HookStack`] that runs several
//! hooks in registration order.
//!
//! # Composition: mergeable patches vs. terminal control actions
//!
//! How a [`HookStack`] combines several hooks' [`Flow`] results depends on the
//! event, and this is the central behavior to understand:
//!
//! - **[`StepEvent::CompletionCall`] — accumulate & merge.** Every hook is
//!   consulted. A hook that returns [`Flow::PatchRequest`] does **not** stop the
//!   others: patches from all hooks are merged in registration order into one
//!   effective patch (see [`RequestPatch`] for the per-field merge rules). This
//!   lets a RAG hook, a tool-policy hook, and a provider-param hook all
//!   contribute to the same turn. [`Flow::Terminate`] stops the stack and is
//!   honored; any other (unsupported) flow stops the stack and fails closed,
//!   discarding the accumulated patch.
//! - **[`StepEvent::ToolCall`] / [`StepEvent::ToolResult`] — chain.** Every hook
//!   is consulted; a [`Flow::RewriteArgs`] / [`Flow::RewriteResult`] does not
//!   stop the others — the rewritten value is threaded into the next hook's
//!   event, so hook *N* observes the value as rewritten by hooks *1..N-1* and may
//!   rewrite further (a redaction hook and a truncation hook compose).
//!   [`Flow::Skip`] / [`Flow::Terminate`] are terminal mid-chain.
//! - **Every other event — first non-[`Continue`](Flow::Continue) wins.** These
//!   are observe-only or recovery events ([`CompletionResponse`](StepEvent::CompletionResponse),
//!   [`ModelTurnFinished`](StepEvent::ModelTurnFinished),
//!   [`InvalidToolCall`](StepEvent::InvalidToolCall), the streamed deltas): the
//!   first hook to return a non-[`Continue`](Flow::Continue) result short-circuits
//!   the rest.
//!
//! **Blind merge.** During accumulation/chaining a hook does *not* see earlier
//! hooks' contributions in its event payload for `CompletionCall` (it sees the
//! agent baseline); for `ToolCall`/`ToolResult` it *does* see the running
//! rewritten value. `CompletionCall` patches are declarative with documented
//! conflict rules, so blind merge is sufficient and keeps [`StepEvent`] `Copy`.
//!
//! **Ordering guidance.** Because [`Flow::Terminate`] short-circuits the stack,
//! register observe-only hooks (telemetry) *before* steering hooks so a later
//! terminate cannot hide the run from them. A [`HookStack`] pushed *as a hook*
//! into another stack composes correctly: it returns its own net flow (a merged
//! patch, a threaded rewrite, or a terminal action) which the outer stack folds
//! in again — nesting never reintroduces short-circuiting on mergeable results.
//!
//! # Why a returned [`Flow`], not a `next()`-style middleware
//!
//! A hook returns a typed [`Flow`] rather than receiving a `next` continuation it
//! must invoke. A `next()`/middleware model — where each layer has to call
//! `next(ctx)` to let the rest of the chain *and* the wrapped action run — carries
//! a well-known footgun: forgetting the call silently disables every downstream
//! hook and the action itself, with no error. The declarative returned-[`Flow`]
//! model makes that impossible: proceeding is the explicit [`Flow::Continue`], and
//! any action an event cannot honor is fail-closed (it terminates the run) rather
//! than silently skipped.
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
//! [`Flow::Continue`]). Every `on_event` now also receives a [`HookContext`]
//! first argument. Attach one or more hooks with `add_hook`.
//!
//! | Old `PromptHook` method | [`StepEvent`] variant | [`Flow`] to return |
//! |---|---|---|
//! | `on_completion_call` | [`CompletionCall`](StepEvent::CompletionCall) `{ prompt, history, turn }` | [`cont`](Flow::cont) / [`patch_request`](Flow::patch_request) / [`terminate`](Flow::terminate) |
//! | `on_completion_response` | [`CompletionResponse`](StepEvent::CompletionResponse) `{ prompt, response }` | [`cont`](Flow::cont) / [`terminate`](Flow::terminate) |
//! | `on_invalid_tool_call` | [`InvalidToolCall`](StepEvent::InvalidToolCall)`(ctx)` | [`fail`](Flow::fail) (default) / [`retry`](Flow::retry) / [`repair`](Flow::repair) / [`skip`](Flow::skip) / [`terminate`](Flow::terminate) |
//! | `on_tool_call` | [`ToolCall`](StepEvent::ToolCall) `{ tool_name, tool_call_id, internal_call_id, args }` | [`cont`](Flow::cont) / [`rewrite_args`](Flow::rewrite_args) / [`skip`](Flow::skip) / [`terminate`](Flow::terminate) |
//! | `on_tool_result` | [`ToolResult`](StepEvent::ToolResult) `{ tool_name, .., result, outcome, extensions }` | [`cont`](Flow::cont) / [`rewrite_result`](Flow::rewrite_result) / [`terminate`](Flow::terminate) |
//! | `on_text_delta` | [`TextDelta`](StepEvent::TextDelta) `{ delta, aggregated }` | [`cont`](Flow::cont) / [`terminate`](Flow::terminate) |
//! | `on_tool_call_delta` | [`ToolCallDelta`](StepEvent::ToolCallDelta) `{ tool_call_id, internal_call_id, tool_name, delta }` | [`cont`](Flow::cont) / [`terminate`](Flow::terminate) |
//! | `on_stream_completion_response_finish` | [`StreamResponseFinish`](StepEvent::StreamResponseFinish) `{ prompt, response }` | [`cont`](Flow::cont) / [`terminate`](Flow::terminate) |
//! | *(new, both surfaces)* | [`ModelTurnFinished`](StepEvent::ModelTurnFinished) `{ turn, content, usage }` | [`cont`](Flow::cont) / [`terminate`](Flow::terminate) |
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
//!
//! # Steering on structured tool outcomes
//!
//! [`StepEvent::ToolResult`] carries a structured
//! [`ToolOutcome`] alongside the model-visible
//! `result`, so a hook can branch on *why* a tool failed — a timeout vs. a 404 —
//! without parsing strings. The motivating case: abort after repeated timeouts,
//! but let a not-found flow back to the model as recoverable feedback.
//!
//! ```rust,ignore
//! use rig_core::agent::{AgentHook, Flow, HookContext, StepEvent};
//! use rig_core::completion::CompletionModel;
//! use rig_core::tool::ToolFailureKind;
//!
//! #[derive(Clone, Default)]
//! struct TimeoutCount(usize);
//!
//! struct OutcomePolicy;
//!
//! impl<M: CompletionModel> AgentHook<M> for OutcomePolicy {
//!     async fn on_event(&self, ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
//!         if let StepEvent::ToolResult { outcome, .. } = event {
//!             // Repeated timeouts abort the run; a 404 does not.
//!             if outcome.is_error_kind(ToolFailureKind::Timeout) {
//!                 let count = ctx.scratchpad().update(|c: &mut TimeoutCount| {
//!                     c.0 += 1;
//!                     c.0
//!                 });
//!                 if count >= 10 {
//!                     return Flow::terminate("aborting after repeated tool timeouts");
//!                 }
//!             }
//!             // `NotFound` falls through to `Flow::cont`: the model sees the
//!             // error text and may try another path.
//!         }
//!         Flow::cont()
//!     }
//! }
//! ```

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::{
    OneOrMany,
    completion::{CompletionModel, Document, Usage},
    json_utils,
    message::{AssistantContent, Message, ToolChoice},
    tool::{ToolCallExtensions, ToolOutcome, ToolResultExtensions},
    wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync},
};

/// Opaque, process-scoped identifier for a single agent run.
///
/// Minted once when a run's [`HookContext`] is created and stable for the whole
/// run, so a hook can correlate every event it observes (across turns, tool
/// calls and streamed deltas) to one run. It is a short URL-safe string from
/// Rig's internal, non-cryptographic id generator — not globally unique across
/// process restarts, and not security-sensitive.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RunId(String);

impl RunId {
    /// Mint a fresh run id.
    pub(crate) fn generate() -> Self {
        Self(crate::id::generate())
    }

    /// The id as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for RunId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// A run-scoped, shared scratchpad passed to every hook via [`HookContext`].
///
/// A type-map with interior mutability: cooperating hooks read and write typed
/// values keyed by type, sharing per-run state (a turn counter, a running
/// budget, a phase flag) without each rolling its own `Arc<Mutex<…>>`. Every
/// hook in a run receives the same [`HookContext`] by shared reference, so they
/// all see one scratchpad (and cloning a `Scratchpad` shares its storage too); a
/// fresh run starts with an empty one.
///
/// Hooks receive `&HookContext` (shared), so every accessor here takes `&self`
/// and mutates through an internal lock. Reads clone the stored value out (the
/// lock cannot hand out a borrow), so store cheaply-cloneable values.
///
/// # Concurrency
///
/// Most events are dispatched sequentially within a run, but at
/// [`tool_concurrency`](crate::agent::AgentRunner::tool_concurrency)` > 1` the
/// [`ToolCall`](StepEvent::ToolCall) / [`ToolResult`](StepEvent::ToolResult)
/// hooks for *different* tools in the same turn may run **concurrently**, all
/// sharing this one scratchpad. Each accessor ([`insert`](Self::insert),
/// [`update`](Self::update), …) is race-free *per operation* (it holds the lock
/// for the whole read-modify-write), but the framework imposes **no
/// deterministic ordering** across those concurrent tool hooks — the order in
/// which two tools' hooks touch the scratchpad depends on tool completion
/// timing. Prefer commutative / idempotent state (a counter, a set union), or
/// key per-tool state by the tool call id / internal call id, rather than
/// relying on the order of concurrent updates.
///
/// # Example
/// ```
/// # use rig_core::agent::hook::Scratchpad;
/// #[derive(Clone, Default)]
/// struct Calls(u32);
///
/// let pad = Scratchpad::default();
/// pad.update(|c: &mut Calls| c.0 += 1);
/// assert_eq!(pad.get::<Calls>().map(|c| c.0), Some(1));
/// ```
#[derive(Clone, Default)]
pub struct Scratchpad {
    // Reuses the tested `ToolCallExtensions` type-map as the storage, wrapped in
    // a shared lock so `&HookContext` hooks can mutate it. Under
    // `tool_concurrency > 1` several tools' `ToolCall`/`ToolResult` hooks may
    // touch this concurrently, so the lock is load-bearing, not decorative.
    inner: Arc<std::sync::Mutex<ToolCallExtensions>>,
    calls: Arc<std::sync::Mutex<std::collections::HashMap<String, ToolCallExtensions>>>,
}

impl Scratchpad {
    fn lock(&self) -> std::sync::MutexGuard<'_, ToolCallExtensions> {
        // A poisoned scratchpad (a hook panicked while holding the lock) should
        // not cascade into cancelling later hooks or the run; recover the guard.
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Insert a typed value, returning the previous value of the same type.
    pub fn insert<T: Clone + WasmCompatSend + WasmCompatSync + 'static>(
        &self,
        val: T,
    ) -> Option<T> {
        self.lock().insert(val)
    }

    /// Get a clone of the stored value of type `T`, if present.
    pub fn get<T: Clone + WasmCompatSend + WasmCompatSync + 'static>(&self) -> Option<T> {
        self.lock().get::<T>().cloned()
    }

    /// Whether a value of type `T` is present.
    pub fn contains<T: WasmCompatSend + WasmCompatSync + 'static>(&self) -> bool {
        self.lock().contains::<T>()
    }

    /// Remove and return the stored value of type `T`, if present.
    pub fn remove<T: Clone + WasmCompatSend + WasmCompatSync + 'static>(&self) -> Option<T> {
        self.lock().remove::<T>()
    }

    /// Read-modify-write the value of type `T` under one lock acquisition,
    /// starting from [`Default`] when absent. The value is stored back and the
    /// closure's return value is returned.
    ///
    /// The whole read-modify-write is atomic (no lost updates), but at
    /// `tool_concurrency > 1` it imposes **no ordering** across concurrent tool
    /// hooks — see the [type-level concurrency note](Scratchpad#concurrency).
    /// This is the race-free way to bump a counter or accumulate:
    /// ```
    /// # use rig_core::agent::hook::Scratchpad;
    /// # #[derive(Clone, Default)] struct Total(u64);
    /// # let pad = Scratchpad::default();
    /// pad.update(|t: &mut Total| t.0 += 10);
    /// ```
    pub fn update<T, R>(&self, f: impl FnOnce(&mut T) -> R) -> R
    where
        T: Clone + Default + WasmCompatSend + WasmCompatSync + 'static,
    {
        let mut guard = self.lock();
        let mut val = guard.remove::<T>().unwrap_or_default();
        let out = f(&mut val);
        guard.insert(val);
        out
    }

    /// Return isolated state for one internal tool-call identifier.
    ///
    /// Different concurrent calls cannot overwrite or consume each other's
    /// values even when they store the same Rust type.
    pub fn for_call(&self, internal_call_id: impl Into<String>) -> CallScratchpad {
        CallScratchpad {
            id: internal_call_id.into(),
            calls: self.calls.clone(),
        }
    }
}

/// Type-keyed hook state isolated to one tool call.
#[derive(Clone, Debug)]
pub struct CallScratchpad {
    id: String,
    calls: Arc<std::sync::Mutex<std::collections::HashMap<String, ToolCallExtensions>>>,
}

impl CallScratchpad {
    fn lock(
        &self,
    ) -> std::sync::MutexGuard<'_, std::collections::HashMap<String, ToolCallExtensions>> {
        self.calls.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Insert a value for this call, returning the previous value of its type.
    pub fn insert<T>(&self, value: T) -> Option<T>
    where
        T: Clone + WasmCompatSend + WasmCompatSync + 'static,
    {
        self.lock()
            .entry(self.id.clone())
            .or_default()
            .insert(value)
    }

    /// Clone a value stored for this call.
    pub fn get<T>(&self) -> Option<T>
    where
        T: Clone + WasmCompatSend + WasmCompatSync + 'static,
    {
        self.lock()
            .get(&self.id)
            .and_then(|state| state.get::<T>().cloned())
    }

    /// Atomically update a value for this call, starting from `Default`.
    pub fn update<T, R>(&self, f: impl FnOnce(&mut T) -> R) -> R
    where
        T: Clone + Default + WasmCompatSend + WasmCompatSync + 'static,
    {
        let mut calls = self.lock();
        let state = calls.entry(self.id.clone()).or_default();
        let mut value = state.remove::<T>().unwrap_or_default();
        let output = f(&mut value);
        state.insert(value);
        output
    }

    /// Remove a value stored for this call.
    pub fn remove<T>(&self) -> Option<T>
    where
        T: Clone + WasmCompatSend + WasmCompatSync + 'static,
    {
        self.lock()
            .get_mut(&self.id)
            .and_then(|state| state.remove::<T>())
    }
}

impl std::fmt::Debug for Scratchpad {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scratchpad")
            .field("entries", &self.lock().len())
            .finish()
    }
}

/// Run-scoped context passed by shared reference to every [`AgentHook::on_event`]
/// call.
///
/// Carries the run's identity and a shared [`Scratchpad`]. It is a *driver*
/// construct built once per run by [`AgentRunner`](crate::agent::AgentRunner);
/// nothing here reaches the sans-IO [`AgentRun`](crate::agent::run::AgentRun)
/// state machine. Hooks hold it by `&`, so all fields are read via accessors and
/// run-scoped mutation goes through [`scratchpad`](Self::scratchpad).
///
/// One `HookContext` is shared by every hook invocation in a run. At
/// [`tool_concurrency`](crate::agent::AgentRunner::tool_concurrency)` > 1` the
/// [`ToolCall`](StepEvent::ToolCall) / [`ToolResult`](StepEvent::ToolResult)
/// hooks for different tools in a turn can run concurrently against this shared
/// context — see the [`Scratchpad` concurrency note](Scratchpad#concurrency) for
/// how to store run-scoped state safely under that concurrency.
#[derive(Debug)]
pub struct HookContext {
    run_id: RunId,
    // Interior-mutable so the driver can advance it each turn while hooks hold a
    // shared `&HookContext`; also the reason the context is `Sync`.
    turn: AtomicUsize,
    is_streaming: bool,
    agent_name: Option<String>,
    scratchpad: Scratchpad,
}

impl HookContext {
    /// Build a fresh run-scoped context. `is_streaming` records which surface is
    /// driving ([`run`](crate::agent::AgentRunner::run) vs.
    /// [`stream`](crate::agent::AgentRunner::stream)).
    #[cfg(test)]
    pub(crate) fn new(is_streaming: bool, agent_name: Option<String>) -> Self {
        Self::with_run_id(RunId::generate(), is_streaming, agent_name)
    }

    pub(crate) fn with_run_id(
        run_id: RunId,
        is_streaming: bool,
        agent_name: Option<String>,
    ) -> Self {
        Self {
            run_id,
            turn: AtomicUsize::new(0),
            is_streaming,
            agent_name,
            scratchpad: Scratchpad::default(),
        }
    }

    /// Record the current one-based model-call index (set by the driver before
    /// each turn), so events that don't carry a turn still see it.
    pub(crate) fn set_turn(&self, turn: usize) {
        self.turn.store(turn, Ordering::Relaxed);
    }

    /// The run's stable identifier.
    pub fn run_id(&self) -> &RunId {
        &self.run_id
    }

    /// The current one-based model-call index (0 before the first turn).
    pub fn turn(&self) -> usize {
        self.turn.load(Ordering::Relaxed)
    }

    /// Whether this run is driven by the streaming surface.
    pub fn is_streaming(&self) -> bool {
        self.is_streaming
    }

    /// The agent's configured name, if any.
    pub fn agent_name(&self) -> Option<&str> {
        self.agent_name.as_deref()
    }

    /// The run-scoped shared scratchpad.
    pub fn scratchpad(&self) -> &Scratchpad {
        &self.scratchpad
    }
}

// `&HookContext` is borrowed across `.await` points in async hook dispatch, so
// on native targets `HookContext` must stay `Sync` (and `Send`). This fails to
// compile if a future change drops the property.
#[cfg(not(target_family = "wasm"))]
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<HookContext>();
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
    /// [`Flow::Continue`], [`Flow::PatchRequest`] (patch this turn's request) and
    /// [`Flow::Terminate`]. Across a [`HookStack`], every hook's
    /// [`PatchRequest`](Flow::PatchRequest) is merged (see the module docs).
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
    /// [`Flow::Continue`] and [`Flow::Terminate`]. The medium-specific
    /// (non-streaming) counterpart of [`ModelTurnFinished`](Self::ModelTurnFinished),
    /// carrying the raw provider response.
    CompletionResponse {
        /// The prompt message for this turn.
        prompt: &'a Message,
        /// The model's completion response.
        response: &'a crate::completion::CompletionResponse<M::Response>,
    },
    /// After a model turn is accepted into the run, on **both** surfaces,
    /// regardless of whether the turn produced text, tool calls, reasoning, or
    /// mixed content. This is the normalized, medium-neutral counterpart of
    /// [`CompletionResponse`](Self::CompletionResponse) (non-streaming) and
    /// [`StreamResponseFinish`](Self::StreamResponseFinish) (streaming) — use it
    /// for telemetry that must fire once per turn everywhere, including a
    /// streamed tool-only turn that fires no `StreamResponseFinish`. Suppressed
    /// for turns recovered by invalid tool-call repair, skip, or retry, and
    /// fired *after* the medium-specific raw event when one fires. Observe-only:
    /// honors [`Flow::Continue`] and [`Flow::Terminate`].
    ModelTurnFinished {
        /// One-based index of this model call within the run.
        turn: usize,
        /// The model's assistant content for this turn — the canonical committed
        /// model output. For an ordinary turn this is exactly what is recorded
        /// into the run. On a structured-output Tool-mode turn that finalizes by
        /// calling the output tool, this is the model-emitted content **including**
        /// that output-tool call; the run then persists the turn as assistant text
        /// (the structured output) with the tool call dropped, so the persisted
        /// message differs from this content.
        content: &'a OneOrMany<AssistantContent>,
        /// Token usage for this turn (zeroed if the provider reported none).
        usage: Usage,
    },
    /// The model emitted a tool call that is unknown or disallowed for this
    /// turn. Honors [`Flow::Fail`] (the default), [`Flow::Retry`],
    /// [`Flow::Repair`], [`Flow::Skip`] and [`Flow::Terminate`];
    /// [`Flow::Continue`] is treated as [`Flow::Fail`].
    InvalidToolCall(&'a InvalidToolCallContext),
    /// Before a tool is executed. Honors [`Flow::Continue`],
    /// [`Flow::RewriteArgs`] (execute the tool with rewritten arguments),
    /// [`Flow::Skip`] (return `reason` as the tool result without executing) and
    /// [`Flow::Terminate`]. Across a [`HookStack`], [`RewriteArgs`](Flow::RewriteArgs)
    /// is chained: `args` reflects prior hooks' rewrites (see the module docs).
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
    /// After a tool has produced a result (or a [`ToolCall`](Self::ToolCall) hook
    /// [skipped](Flow::Skip) it). Honors [`Flow::Continue`],
    /// [`Flow::RewriteResult`] (substitute the result the model sees) and
    /// [`Flow::Terminate`].
    ///
    /// `result` is the model-visible output, and `outcome` / `extensions` are the
    /// **structured** execution result — the machine-visible half a hook inspects
    /// without parsing `result`. `outcome` distinguishes success from a classified
    /// [`ToolFailure`](crate::tool::ToolFailure) (timeout, not-found, …), a
    /// [`Skipped`](crate::tool::ToolOutcome::Skipped) call, or a
    /// [`Denied`](crate::tool::ToolOutcome::Denied) one; `extensions` carries
    /// provider/application metadata the tool attached that is never sent to the
    /// model.
    ///
    /// For the first hook, `result` is the tool's actual output and `outcome` its
    /// raw structured outcome; across a [`HookStack`],
    /// [`RewriteResult`](Flow::RewriteResult) is chained so a later hook sees the
    /// prior hook's replacement in `result`. A rewrite changes only `result` (the
    /// model-visible text) — `outcome` and `extensions` are the tool's raw
    /// structured result throughout, so a redaction hook cannot mask the true
    /// outcome from a later policy hook (see the module docs).
    ToolResult {
        /// Name of the tool that was called.
        tool_name: &'a str,
        /// Provider-supplied tool call ID, when available.
        tool_call_id: Option<&'a str>,
        /// Internal Rig call ID correlating this call's events.
        internal_call_id: &'a str,
        /// JSON arguments for the call.
        args: &'a str,
        /// The model-visible tool result. Reflects any earlier hook's
        /// [`RewriteResult`](Flow::RewriteResult); the first hook sees the tool's
        /// actual output.
        result: &'a str,
        /// The structured outcome of the execution (success / classified error /
        /// skipped / denied). The raw outcome, unaffected by `RewriteResult`.
        outcome: &'a ToolOutcome,
        /// Metadata the tool attached to its result, never sent to the model.
        extensions: &'a ToolResultExtensions,
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
    /// not fire it. For a per-turn event that fires on *every* turn on both
    /// surfaces, use [`ModelTurnFinished`](Self::ModelTurnFinished). Honors
    /// [`Flow::Continue`] and [`Flow::Terminate`].
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
    /// [`StepEvent::ModelTurnFinished`].
    ModelTurnFinished,
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
            StepEvent::ModelTurnFinished { .. } => StepEventKind::ModelTurnFinished,
            StepEvent::InvalidToolCall(_) => StepEventKind::InvalidToolCall,
            StepEvent::ToolCall { .. } => StepEventKind::ToolCall,
            StepEvent::ToolResult { .. } => StepEventKind::ToolResult,
            StepEvent::TextDelta { .. } => StepEventKind::TextDelta,
            StepEvent::ToolCallDelta { .. } => StepEventKind::ToolCallDelta,
            StepEvent::StreamResponseFinish { .. } => StepEventKind::StreamResponseFinish,
        }
    }
}

/// A partial patch over the model request for a single turn, returned by a hook
/// via [`Flow::PatchRequest`] on a [`StepEvent::CompletionCall`] event.
///
/// Every field is optional: a `Some` value overrides the agent's configured
/// value for this turn, a `None` value inherits it. The patch is **per-turn and
/// non-sticky** — it never changes the agent's baseline, so the next turn
/// re-fires [`CompletionCall`](StepEvent::CompletionCall) and resolves from the
/// baseline again.
///
/// # Merge behavior
///
/// Two kinds of merge apply. When several hooks in a [`HookStack`] each return a
/// patch, they are combined **hook ⊕ hook in registration order** with these
/// per-field rules; the effective patch is then applied **patch → baseline**.
///
/// | Field | hook ⊕ hook (registration order) | patch → baseline |
/// |---|---|---|
/// | `extra_context` | append (earlier hooks' docs first) | append after static + dynamic context |
/// | `additional_params` | shallow-merge top-level keys, later hook wins | shallow-merge onto baseline params |
/// | `preamble` | last writer wins (warns on conflict) | replaces |
/// | `temperature`, `max_tokens`, `tool_choice` | last writer wins (warns on conflict) | replaces |
/// | `active_tools` | set **intersection** (warns when empty) | narrows the advertised set |
/// | `history` | last writer wins (warns on conflict) | replaces the messages sent this turn |
///
/// `active_tools` intersects rather than last-writer-wins because it is an
/// allow-list guardrail: two narrowing hooks must compose as *narrowing*. All
/// last-writer-wins conflicts emit a `tracing::warn!` so composition stays
/// debuggable — additive guidance belongs in `extra_context` documents, not in
/// preamble concatenation.
///
/// Build one with the setters:
///
/// ```rust,ignore
/// Flow::patch_request(
///     RequestPatch::new()
///         .tool_choice(ToolChoice::Required)
///         .active_tools(["search"])
///         .temperature(0.0),
/// )
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct RequestPatch {
    /// Override the system prompt / preamble for this turn.
    pub preamble: Option<String>,
    /// Override the sampling temperature for this turn.
    pub temperature: Option<f64>,
    /// Override the max output tokens for this turn.
    pub max_tokens: Option<u64>,
    /// Override the tool choice for this turn.
    pub tool_choice: Option<ToolChoice>,
    /// Restrict the advertised tools to this allow-list (by name) for this turn.
    /// `Some(vec![])` advertises no executable tools; `None` keeps the full set.
    pub active_tools: Option<Vec<String>>,
    /// Provider-passthrough params shallow-merged onto the agent's for this turn.
    pub additional_params: Option<serde_json::Value>,
    /// Extra context documents appended (after static and dynamic context) for
    /// this turn only. The passive-RAG injection point.
    pub extra_context: Vec<Document>,
    /// Replace the prior chat history sent to the provider **this turn only**.
    /// The persisted transcript and the run state are untouched, and RAG's query
    /// text still derives from the original prompt/history — this changes only
    /// what messages are sent. `None` sends the real history. The enabling
    /// primitive for context-window compaction / summarization middleware.
    pub history: Option<Vec<Message>>,
}

/// Last-writer-wins merge for a scalar patch field, warning on a real conflict.
fn merge_last_wins<T>(earlier: Option<T>, later: Option<T>, field: &str) -> Option<T> {
    match (earlier, later) {
        (Some(_), Some(l)) => {
            tracing::warn!(
                patch_field = field,
                "two hooks set `{field}` on the same turn; the later hook wins"
            );
            Some(l)
        }
        (earlier, later) => later.or(earlier),
    }
}

impl RequestPatch {
    /// An empty patch — a no-op, identical to returning [`Flow::cont`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the system prompt / preamble for this turn.
    pub fn preamble(mut self, preamble: impl Into<String>) -> Self {
        self.preamble = Some(preamble.into());
        self
    }

    /// Override the sampling temperature for this turn.
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Override the max output tokens for this turn.
    pub fn max_tokens(mut self, max_tokens: u64) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Override the tool choice for this turn.
    ///
    /// Not every provider honors `tool_choice`: some in-core providers (e.g.
    /// Ollama, Hyperbolic, Mira, Perplexity) ignore it and log a warning, so
    /// forcing a tool this way is a no-op there. A choice a provider cannot
    /// represent (e.g. a multi-name [`ToolChoice::Specific`] on Anthropic, which
    /// forces a single tool) surfaces as a request error rather than being
    /// silently downgraded.
    pub fn tool_choice(mut self, tool_choice: ToolChoice) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    /// Restrict the advertised tools to this allow-list (by name) for this turn.
    ///
    /// This narrows the executable tool set, so it composes with `tool_choice`:
    /// if the effective tool choice is a [`ToolChoice::Specific`] naming a tool
    /// that `active_tools` filters out (e.g. the agent's baseline choice is
    /// inherited because this patch didn't set its own), the request fails
    /// closed with a request error rather than silently forcing a dropped tool.
    /// When narrowing the set, set a compatible `tool_choice` in the same patch.
    pub fn active_tools<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.active_tools = Some(names.into_iter().map(Into::into).collect());
        self
    }

    /// Shallow-merge these provider-passthrough params onto the agent's for this
    /// turn.
    pub fn additional_params(mut self, additional_params: serde_json::Value) -> Self {
        self.additional_params = Some(additional_params);
        self
    }

    /// Append extra context documents for this turn (the passive-RAG injection
    /// point). Documents are appended after the agent's static and dynamic
    /// (vector-store) context, in the order added.
    pub fn extra_context<I>(mut self, docs: I) -> Self
    where
        I: IntoIterator<Item = Document>,
    {
        self.extra_context.extend(docs);
        self
    }

    /// Append a single extra context document for this turn.
    pub fn context(mut self, doc: Document) -> Self {
        self.extra_context.push(doc);
        self
    }

    /// Replace the prior chat history sent to the provider **this turn only**.
    /// The persisted transcript is untouched; RAG query text still derives from
    /// the original history. Use for context-window compaction / summarization.
    pub fn history<I>(mut self, history: I) -> Self
    where
        I: IntoIterator<Item = Message>,
    {
        self.history = Some(history.into_iter().collect());
        self
    }

    /// Whether this patch has no effect (all fields unset).
    pub(crate) fn is_empty(&self) -> bool {
        self.preamble.is_none()
            && self.temperature.is_none()
            && self.max_tokens.is_none()
            && self.tool_choice.is_none()
            && self.active_tools.is_none()
            && self.additional_params.is_none()
            && self.extra_context.is_empty()
            && self.history.is_none()
    }

    /// Merge a later hook's patch onto this one (registration order: `self` is
    /// the accumulated earlier hooks, `later` is the next hook). See the struct
    /// docs for the per-field rules.
    pub(crate) fn merge(mut self, later: RequestPatch) -> RequestPatch {
        // extra_context: append (earlier hooks' documents first).
        self.extra_context.extend(later.extra_context);

        // additional_params: shallow-merge when both are objects (later wins per
        // key), otherwise the later non-None value wins wholesale — mirroring the
        // patch → baseline behavior in request assembly.
        self.additional_params = match (self.additional_params.take(), later.additional_params) {
            (Some(base), Some(patch)) if base.is_object() && patch.is_object() => {
                Some(json_utils::merge(base, patch))
            }
            (base, patch) => patch.or(base),
        };

        // Scalars + preamble + history: last writer wins, warn on real conflict.
        self.preamble = merge_last_wins(self.preamble, later.preamble, "preamble");
        self.temperature = merge_last_wins(self.temperature, later.temperature, "temperature");
        self.max_tokens = merge_last_wins(self.max_tokens, later.max_tokens, "max_tokens");
        self.tool_choice = merge_last_wins(self.tool_choice, later.tool_choice, "tool_choice");
        self.history = merge_last_wins(self.history, later.history, "history");

        // active_tools: set intersection (two narrowing guardrails compose as
        // narrowing). One-sided keeps the present allow-list.
        self.active_tools = match (self.active_tools.take(), later.active_tools) {
            (Some(earlier), Some(later)) => {
                let later_set: std::collections::BTreeSet<&String> = later.iter().collect();
                let intersection: Vec<String> = earlier
                    .into_iter()
                    .filter(|name| later_set.contains(name))
                    .collect();
                if intersection.is_empty() {
                    tracing::warn!(
                        "two hooks' `active_tools` allow-lists have an empty intersection; \
                         no executable tools will be advertised this turn"
                    );
                }
                Some(intersection)
            }
            (earlier, later) => earlier.or(later),
        };

        self
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
///
/// `Flow` is `PartialEq` but not `Eq`, because [`Flow::PatchRequest`] carries a
/// [`RequestPatch`] whose `temperature` is an `f64`.
#[derive(Debug, Clone, PartialEq)]
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
        /// The message returned to the model in place of the tool result. It is
        /// delivered verbatim, so it doubles as a prompt: state that the tool did
        /// not run and, unless you want the model to try again, tell it not to
        /// retry — a bare `"denied"` often makes the model re-emit the same call.
        reason: String,
    },
    /// [`StepEvent::ToolCall`] only: rewrite the tool-call arguments, then
    /// execute the tool with the replacement. This is the steering action for
    /// guardrails that normalize, clamp, redirect, or inject scoped parameters
    /// before a tool runs.
    ///
    /// The rewritten arguments are what the tool is invoked with, what the
    /// following [`StepEvent::ToolResult`] reports, and what the
    /// `gen_ai.tool.call.arguments` span field records.
    ///
    /// This rewrites only what the tool *executes against*, not the model's
    /// transcript: the assistant message that recorded the original tool call is
    /// unchanged and keeps the model's original arguments. It is therefore an
    /// execution-args rewrite (inject defaults, clamp a range, redirect a path),
    /// **not** a history redactor — it does not scrub a value the model already
    /// emitted from the conversation.
    ///
    /// Across a [`HookStack`], rewrites **chain**: the rewritten arguments are
    /// threaded into the next hook's [`ToolCall`](StepEvent::ToolCall) event, so
    /// several hooks can each refine the arguments in registration order.
    RewriteArgs {
        /// The JSON arguments the tool is invoked with, in place of the ones the
        /// model emitted.
        args: serde_json::Value,
    },
    /// [`StepEvent::ToolResult`] only: replace the tool's result with this string
    /// before the model sees it. The post-execution counterpart of
    /// [`RewriteArgs`](Flow::RewriteArgs) — for guardrails that redact, truncate,
    /// or normalize a tool's output.
    ///
    /// The replacement is what the model receives as the tool result and what the
    /// `gen_ai.tool.call.result` span field records. As with
    /// [`RewriteArgs`](Flow::RewriteArgs), this changes only what the model
    /// *sees*: the tool still ran and produced its real output (which the first
    /// hook's [`ToolResult`](StepEvent::ToolResult) event observed before this
    /// replacement is applied). It does not scrub the tool's output from logs.
    ///
    /// The replacement is delivered to the model verbatim — it is not re-parsed
    /// as structured/multimodal tool output, so a JSON-shaped replacement reaches
    /// the model as literal text.
    ///
    /// Across a [`HookStack`], rewrites **chain**: the replacement is threaded
    /// into the next hook's [`ToolResult`](StepEvent::ToolResult) event, so a
    /// redaction hook and a truncation hook can compose in registration order.
    RewriteResult {
        /// The result delivered to the model in place of the tool's actual
        /// output.
        result: String,
    },
    /// [`StepEvent::CompletionCall`] only: patch fields of the model request for
    /// this turn before it is sent. The per-turn request-steering action — for
    /// hooks that adjust the system prompt, sampling, tool choice, the advertised
    /// tool set, or inject context documents from run state (force a tool on the
    /// first turn, lower the temperature on a critical step, add RAG context).
    ///
    /// The patch is partial ([`RequestPatch`]): each set field replaces (or, for
    /// `additional_params`/`extra_context`, merges onto) the agent's configured
    /// value; unset fields are inherited. It applies to *this turn only* and does
    /// not change the agent's baseline — the next turn re-fires
    /// [`CompletionCall`](StepEvent::CompletionCall) and re-resolves from it.
    ///
    /// Across a [`HookStack`], patches from all hooks **accumulate** and merge in
    /// registration order (see [`RequestPatch`] and the module docs); this action
    /// therefore does *not* short-circuit later hooks.
    PatchRequest {
        /// The partial request patch applied to this turn.
        patch: RequestPatch,
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
    ///
    /// `reason` is delivered to the model verbatim as the tool result, so it
    /// doubles as a prompt — tell the model the tool did not run and whether to
    /// retry, or it may re-emit the identical call:
    ///
    /// ```rust,ignore
    /// Flow::skip("Not executed (denied by policy). Do not retry unless the user asks.")
    /// ```
    pub fn skip(reason: impl Into<String>) -> Self {
        Self::Skip {
            reason: reason.into(),
        }
    }

    /// Rewrite a tool call's arguments, then execute the tool with the
    /// replacement (tool calls only).
    ///
    /// Accepts anything convertible into a [`serde_json::Value`] — most often
    /// the [`serde_json::json!`] macro or a value built from the parsed original
    /// arguments. To rewrite from a typed value instead, use
    /// [`try_rewrite_args`](Flow::try_rewrite_args).
    ///
    /// ```rust,ignore
    /// // Inject a scoped parameter the model never sees, leaving the rest intact.
    /// let mut args: serde_json::Value = serde_json::from_str(emitted_args)?;
    /// args["account_id"] = serde_json::json!(session.account_id);
    /// Flow::rewrite_args(args)
    /// ```
    pub fn rewrite_args(args: impl Into<serde_json::Value>) -> Self {
        Self::RewriteArgs { args: args.into() }
    }

    /// Rewrite a tool call's arguments from a serializable value (tool calls
    /// only), serializing it to JSON.
    ///
    /// This is the typed convenience over [`rewrite_args`](Flow::rewrite_args)
    /// for callers that hold a Rust args struct. It only fails if the value
    /// cannot be serialized to JSON; a hook typically maps that error to
    /// [`Flow::terminate`]:
    ///
    /// ```rust,ignore
    /// Flow::try_rewrite_args(&new_args).unwrap_or_else(|e| Flow::terminate(e.to_string()))
    /// ```
    pub fn try_rewrite_args<T: serde::Serialize>(value: &T) -> Result<Self, serde_json::Error> {
        Ok(Self::RewriteArgs {
            args: serde_json::to_value(value)?,
        })
    }

    /// Replace a tool's result with `result` before the model sees it (tool
    /// results only).
    ///
    /// The post-execution counterpart of [`rewrite_args`](Flow::rewrite_args),
    /// for guardrails that redact, truncate, or normalize a tool's output:
    ///
    /// ```rust,ignore
    /// // Redact a secret from the tool output before it reaches the model.
    /// Flow::rewrite_result(redact(tool_output))
    /// ```
    pub fn rewrite_result(result: impl Into<String>) -> Self {
        Self::RewriteResult {
            result: result.into(),
        }
    }

    /// Patch fields of the model request for this turn (completion calls only).
    /// See [`RequestPatch`] for the partial-patch, per-turn, mergeable semantics.
    pub fn patch_request(patch: RequestPatch) -> Self {
        Self::PatchRequest { patch }
    }

    /// Fail fast on an invalid tool call (the default).
    pub fn fail() -> Self {
        Self::Fail
    }

    /// Retry the model turn with corrective feedback (invalid tool calls only).
    ///
    /// A common recovery is to let the model self-correct by naming the valid
    /// tools, built from the diagnostics in [`InvalidToolCallContext`]:
    ///
    /// ```rust,ignore
    /// // On the `StepEvent::InvalidToolCall(ctx)` arm of `on_event`:
    /// Flow::retry(format!(
    ///     "`{}` is not a valid tool. Call one of: [{}].",
    ///     ctx.tool_name,
    ///     ctx.available_tools.join(", "),
    /// ))
    /// ```
    ///
    /// Without such a hook the invalid-call default stays fail-closed
    /// ([`Flow::Continue`] is treated as [`Flow::fail`]).
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

/// A per-run hook that observes and steers an agent run.
///
/// Implement [`on_event`](AgentHook::on_event) and match on the [`StepEvent`]
/// variants you care about; every other event falls through to the default
/// [`Flow::Continue`]. Hooks must be cheap to share (`Clone` is not required —
/// hooks are held behind an `Arc` once registered).
///
/// # Example
/// ```rust,ignore
/// use rig_core::agent::{AgentHook, Flow, HookContext, StepEvent};
/// use rig_core::completion::CompletionModel;
///
/// #[derive(Clone)]
/// struct Logger;
///
/// impl<M: CompletionModel> AgentHook<M> for Logger {
///     async fn on_event(&self, ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
///         if let StepEvent::ToolCall { tool_name, args, .. } = event {
///             println!("[run {}] calling {tool_name}({args})", ctx.run_id());
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
    /// [`observes`](Self::observes)). Receives the run-scoped [`HookContext`] and
    /// the [`StepEvent`]. The default implementation is a no-op: it ignores every
    /// event and returns [`Flow::Continue`]. It does **not** narrow
    /// [`observes`](Self::observes) (which defaults to `true`), so a hook that
    /// takes this default is still dispatched every event — override `observes`
    /// to skip the high-frequency delta events. (The `()` no-op hook overrides
    /// `observes` to `false`, so the runner skips dispatching those delta events
    /// to it; it still receives, and returns [`Flow::Continue`] for, every other
    /// event.)
    fn on_event(
        &self,
        ctx: &HookContext,
        event: StepEvent<'_, M>,
    ) -> impl Future<Output = Flow> + WasmCompatSend {
        let _ = (ctx, event);
        async { Flow::Continue }
    }

    /// Resolve a [`ToolCall`](StepEvent::ToolCall) for this hook, returning its
    /// [`Flow`] plus any tool-argument rewrite that must be **salvaged** when the
    /// hook short-circuits — so a nested [`HookStack`] never loses an inner
    /// [`Flow::RewriteArgs`] behind a later inner
    /// [`Flow::Skip`]/[`Flow::Terminate`].
    ///
    /// The default — correct for any leaf hook — dispatches the `ToolCall` event
    /// to [`on_event`](Self::on_event) and reports **no** salvaged rewrite: a
    /// single hook returns exactly one [`Flow`], so it can either rewrite (via
    /// [`Flow::RewriteArgs`]) or short-circuit, never both. [`HookStack`]
    /// overrides this to compose its members' resolutions, preserving an inner
    /// rewrite across a short-circuit. This is an internal composition hook;
    /// implementing it is only necessary for a custom composite hook that wraps
    /// other hooks and needs the same rewrite-preserving behavior.
    #[doc(hidden)]
    fn resolve_tool_call(
        &self,
        ctx: &HookContext,
        tool_name: &str,
        tool_call_id: Option<&str>,
        internal_call_id: &str,
        args: &str,
    ) -> impl Future<Output = (Flow, Option<serde_json::Value>)> + WasmCompatSend {
        async move {
            let flow = self
                .on_event(
                    ctx,
                    StepEvent::ToolCall {
                        tool_name,
                        tool_call_id,
                        internal_call_id,
                        args,
                    },
                )
                .await;
            (flow, None)
        }
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
impl<M> AgentHook<M> for ()
where
    M: CompletionModel,
{
    /// Observe nothing, so the runner skips building/dispatching the
    /// high-frequency streaming delta events (`TextDelta` / `ToolCallDelta`,
    /// the only events gated on `observes`) for a `()` hook.
    fn observes(&self, _kind: StepEventKind) -> bool {
        false
    }
}

/// Object-safe shim over [`AgentHook`] so a [`HookStack`] can hold a
/// heterogeneous list of hooks behind `Arc`.
trait DynAgentHook<M>: WasmCompatSend + WasmCompatSync
where
    M: CompletionModel,
{
    fn on_event_boxed<'a>(
        &'a self,
        ctx: &'a HookContext,
        event: StepEvent<'a, M>,
    ) -> WasmBoxedFuture<'a, Flow>
    where
        M: 'a;

    /// Object-safe [`AgentHook::resolve_tool_call`]. Preserves an inner
    /// [`Flow::RewriteArgs`] across a short-circuit so nested [`HookStack`]s
    /// compose correctly.
    fn resolve_tool_call_boxed<'a>(
        &'a self,
        ctx: &'a HookContext,
        tool_name: &'a str,
        tool_call_id: Option<&'a str>,
        internal_call_id: &'a str,
        args: &'a str,
    ) -> WasmBoxedFuture<'a, (Flow, Option<serde_json::Value>)>
    where
        M: 'a;

    fn observes_dyn(&self, kind: StepEventKind) -> bool;
}

impl<M, H> DynAgentHook<M> for H
where
    M: CompletionModel,
    H: AgentHook<M>,
{
    fn on_event_boxed<'a>(
        &'a self,
        ctx: &'a HookContext,
        event: StepEvent<'a, M>,
    ) -> WasmBoxedFuture<'a, Flow>
    where
        M: 'a,
    {
        Box::pin(self.on_event(ctx, event))
    }

    fn resolve_tool_call_boxed<'a>(
        &'a self,
        ctx: &'a HookContext,
        tool_name: &'a str,
        tool_call_id: Option<&'a str>,
        internal_call_id: &'a str,
        args: &'a str,
    ) -> WasmBoxedFuture<'a, (Flow, Option<serde_json::Value>)>
    where
        M: 'a,
    {
        Box::pin(self.resolve_tool_call(ctx, tool_name, tool_call_id, internal_call_id, args))
    }

    fn observes_dyn(&self, kind: StepEventKind) -> bool {
        self.observes(kind)
    }
}

/// An ordered list of hooks run as one hook.
///
/// Each hook is consulted in registration order. How their [`Flow`] results
/// combine depends on the event (see the [module docs](self)):
/// [`CompletionCall`](StepEvent::CompletionCall) patches **accumulate**;
/// [`ToolCall`](StepEvent::ToolCall) / [`ToolResult`](StepEvent::ToolResult)
/// rewrites **chain**; every other event uses **first non-[`Continue`](Flow::Continue)
/// wins**. Because the runner is fail-closed, a non-`Continue` action always
/// takes effect or terminates the run — it is never silently ignored. An empty
/// stack is the no-op hook and [`observes`](HookStack::observes) nothing, so the
/// runner skips event dispatch for it entirely.
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
    /// Compose the stack's members' [`ToolCall`](StepEvent::ToolCall)
    /// resolutions, threading tool-arg rewrites through the chain **and**
    /// preserving them across a short-circuit — including for a member that is
    /// itself a [`HookStack`], which is why members are consulted via
    /// [`resolve_tool_call`](AgentHook::resolve_tool_call) rather than
    /// [`on_event`](AgentHook::on_event) (the latter can only return a single
    /// [`Flow`], losing an inner rewrite behind an inner `Skip`/`Terminate`).
    ///
    /// When the chain proceeds, any rewrite is carried by the returned [`Flow`]
    /// itself ([`RewriteArgs`](Flow::RewriteArgs) for a rewriting chain,
    /// [`Continue`](Flow::Continue) otherwise) and the second element is `None`.
    /// When a member short-circuits with [`Flow::Skip`] / [`Flow::Terminate`] (or
    /// a fail-closed action), that action is returned in the first element while
    /// the accumulated rewrite is salvaged into the second element, so the caller
    /// (`run_single_tool`) can still report the rewritten args on the resulting
    /// [`ToolResult`](StepEvent::ToolResult) event and in tracing rather than
    /// leaking the model's original (pre-rewrite) args. The two are therefore
    /// mutually exclusive: the [`Flow`] is [`RewriteArgs`](Flow::RewriteArgs) only
    /// when the second element is `None`.
    async fn resolve_tool_call(
        &self,
        ctx: &HookContext,
        tool_name: &str,
        tool_call_id: Option<&str>,
        internal_call_id: &str,
        args: &str,
    ) -> (Flow, Option<serde_json::Value>) {
        let mut effective: Option<serde_json::Value> = None;
        for hook in &self.hooks {
            let rewritten = effective.as_ref().map(json_utils::value_to_json_string);
            let args_for_hook = rewritten.as_deref().unwrap_or(args);
            let (flow, salvaged) = hook
                .resolve_tool_call_boxed(
                    ctx,
                    tool_name,
                    tool_call_id,
                    internal_call_id,
                    args_for_hook,
                )
                .await;
            // A member (e.g. a nested `HookStack`) may have rewritten the args
            // before short-circuiting; adopt that rewrite so it is not lost.
            if let Some(rewrite) = salvaged {
                effective = Some(rewrite);
            }
            match flow {
                Flow::Continue => {}
                Flow::RewriteArgs { args } => effective = Some(args),
                // A short-circuit drops the accumulated rewrite from the returned
                // flow, so salvage it in the second element for the caller.
                other => return (other, effective),
            }
        }
        // The chain proceeded: surface any rewrite through the flow itself.
        match effective {
            Some(args) => (Flow::RewriteArgs { args }, None),
            None => (Flow::Continue, None),
        }
    }

    async fn on_event(&self, ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
        match event {
            // Accumulate mergeable request patches from every hook (registration
            // order); short-circuit on `Terminate` or any flow the event cannot
            // honor (fail-closed downstream, discarding the accumulated patch).
            // Hooks see the agent baseline, not earlier hooks' patches (blind
            // merge), which keeps `StepEvent` `Copy`.
            StepEvent::CompletionCall { .. } => {
                let mut merged: Option<RequestPatch> = None;
                for hook in &self.hooks {
                    match hook.on_event_boxed(ctx, event).await {
                        Flow::Continue => {}
                        Flow::PatchRequest { patch } => {
                            merged = Some(match merged {
                                Some(acc) => acc.merge(patch),
                                None => patch,
                            });
                        }
                        other => return other,
                    }
                }
                match merged {
                    Some(patch) if !patch.is_empty() => Flow::PatchRequest { patch },
                    _ => Flow::Continue,
                }
            }
            // Chain tool-arg rewrites: thread the effective arguments through
            // each hook so a later hook observes (and may further rewrite) the
            // value produced by earlier hooks. A proceeding chain surfaces the
            // rewrite as `RewriteArgs`; `Skip`/`Terminate` are terminal and any
            // other flow is returned for fail-closed handling. The salvaged
            // rewrite (second element) matters only to `run_single_tool`, which
            // must report it on a short-circuited `ToolResult`; it is dropped
            // here (this result is observe-only).
            StepEvent::ToolCall {
                tool_name,
                tool_call_id,
                internal_call_id,
                args,
            } => {
                self.resolve_tool_call(ctx, tool_name, tool_call_id, internal_call_id, args)
                    .await
                    .0
            }
            // Chain tool-result rewrites: thread the effective (model-visible)
            // result through each hook (the first hook sees the tool's real
            // output). The structured `outcome`/`extensions` are the tool's raw
            // result and are passed unchanged to every hook — a rewrite alters
            // only the model-visible text, never the outcome a later policy sees.
            StepEvent::ToolResult {
                tool_name,
                tool_call_id,
                internal_call_id,
                args,
                result,
                outcome,
                extensions,
            } => {
                let mut effective: Option<String> = None;
                for hook in &self.hooks {
                    let result_for_hook = effective.as_deref().unwrap_or(result);
                    let per_hook = StepEvent::ToolResult {
                        tool_name,
                        tool_call_id,
                        internal_call_id,
                        args,
                        result: result_for_hook,
                        outcome,
                        extensions,
                    };
                    match hook.on_event_boxed(ctx, per_hook).await {
                        Flow::Continue => {}
                        Flow::RewriteResult { result } => effective = Some(result),
                        other => return other,
                    }
                }
                match effective {
                    Some(result) => Flow::RewriteResult { result },
                    None => Flow::Continue,
                }
            }
            // Observe-only / recovery events: first non-`Continue` wins.
            _ => {
                for hook in &self.hooks {
                    match hook.on_event_boxed(ctx, event).await {
                        Flow::Continue => {}
                        other => return other,
                    }
                }
                Flow::Continue
            }
        }
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

    use super::{
        AgentHook, Flow, HookContext, HookStack, RequestPatch, Scratchpad, StepEvent, StepEventKind,
    };
    use crate::test_utils::MockCompletionModel;

    type M = MockCompletionModel;

    fn ctx() -> HookContext {
        HookContext::new(false, Some("test-agent".to_string()))
    }

    /// Pushes its label when invoked and returns `Continue` or `Terminate`.
    struct Recorder {
        label: u32,
        log: Arc<Mutex<Vec<u32>>>,
        stop: bool,
    }

    impl AgentHook<M> for Recorder {
        async fn on_event(&self, _ctx: &HookContext, _event: StepEvent<'_, M>) -> Flow {
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
        async fn on_event(&self, _ctx: &HookContext, _event: StepEvent<'_, M>) -> Flow {
            Flow::cont()
        }

        fn observes(&self, kind: StepEventKind) -> bool {
            kind == self.0
        }
    }

    /// A hook that returns a fixed patch on `CompletionCall`, and records its
    /// label so we can prove every hook ran.
    struct Patcher {
        label: u32,
        log: Arc<Mutex<Vec<u32>>>,
        patch: RequestPatch,
    }

    impl AgentHook<M> for Patcher {
        async fn on_event(&self, _ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
            self.log.lock().expect("log").push(self.label);
            if matches!(event, StepEvent::CompletionCall { .. }) {
                Flow::patch_request(self.patch.clone())
            } else {
                Flow::cont()
            }
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

    fn completion_call_event() -> StepEvent<'static, M> {
        static PROMPT: std::sync::OnceLock<crate::message::Message> = std::sync::OnceLock::new();
        let prompt = PROMPT.get_or_init(|| crate::message::Message::user("hi"));
        StepEvent::CompletionCall {
            prompt,
            history: &[],
            turn: 1,
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

        let flow = stack.on_event(&ctx(), tool_call_event()).await;

        assert!(matches!(flow, Flow::Continue));
        assert_eq!(*log.lock().expect("log"), vec![1, 2]);
    }

    #[tokio::test]
    async fn first_terminate_short_circuits_on_chained_tool_call() {
        // For a tool-call (a chained event), `Terminate` is terminal mid-chain,
        // so a later hook must not run once an earlier hook terminates.
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

        let flow = stack.on_event(&ctx(), tool_call_event()).await;

        assert!(matches!(flow, Flow::Terminate { .. }));
        assert_eq!(
            *log.lock().expect("log"),
            vec![1],
            "a later hook must not run after an earlier hook terminates"
        );
    }

    #[tokio::test]
    async fn first_terminate_short_circuits_on_observe_only_events() {
        // For an observe-only event (the `_ =>` first-non-`Continue`-wins arm,
        // here a `TextDelta`), the first hook to terminate must short-circuit the
        // rest — this exercises the arm the chained tool-call test does not.
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

        let flow = stack
            .on_event(
                &ctx(),
                StepEvent::TextDelta {
                    delta: "hi",
                    aggregated: "hi",
                },
            )
            .await;

        assert!(matches!(flow, Flow::Terminate { .. }));
        assert_eq!(
            *log.lock().expect("log"),
            vec![1],
            "a later hook must not run after an earlier hook terminates an observe-only event"
        );
    }

    #[tokio::test]
    async fn completion_call_patches_accumulate_and_consult_every_hook() {
        // The core composability fix: a patch from hook 1 must NOT skip hook 2.
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut stack = HookStack::<M>::with(Patcher {
            label: 1,
            log: log.clone(),
            patch: RequestPatch::new().temperature(0.1),
        });
        stack.push(Patcher {
            label: 2,
            log: log.clone(),
            patch: RequestPatch::new().max_tokens(256),
        });

        let flow = stack.on_event(&ctx(), completion_call_event()).await;

        assert_eq!(
            *log.lock().expect("log"),
            vec![1, 2],
            "both hooks must run; a mergeable patch does not short-circuit"
        );
        match flow {
            Flow::PatchRequest { patch } => {
                assert_eq!(patch.temperature, Some(0.1));
                assert_eq!(patch.max_tokens, Some(256));
            }
            other => panic!("expected a merged PatchRequest, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn completion_call_terminate_short_circuits_and_discards_patch() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut stack = HookStack::<M>::with(Patcher {
            label: 1,
            log: log.clone(),
            patch: RequestPatch::new().temperature(0.1),
        });
        // A terminating recorder in the middle.
        stack.push(Recorder {
            label: 2,
            log: log.clone(),
            stop: true,
        });
        stack.push(Patcher {
            label: 3,
            log: log.clone(),
            patch: RequestPatch::new().max_tokens(256),
        });

        let flow = stack.on_event(&ctx(), completion_call_event()).await;

        assert!(matches!(flow, Flow::Terminate { .. }));
        assert_eq!(
            *log.lock().expect("log"),
            vec![1, 2],
            "hook 3 must not run after a terminate"
        );
    }

    #[tokio::test]
    async fn nested_stack_composes_patches_without_inner_short_circuit() {
        // A HookStack pushed as a hook must not reintroduce short-circuiting:
        // the inner stack returns its own merged patch, which the outer stack
        // merges again.
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut inner = HookStack::<M>::with(Patcher {
            label: 1,
            log: log.clone(),
            patch: RequestPatch::new().temperature(0.2),
        });
        inner.push(Patcher {
            label: 2,
            log: log.clone(),
            patch: RequestPatch::new().max_tokens(128),
        });

        let mut outer = HookStack::<M>::with(inner);
        outer.push(Patcher {
            label: 3,
            log: log.clone(),
            patch: RequestPatch::new().preamble("outer"),
        });

        let flow = outer.on_event(&ctx(), completion_call_event()).await;

        assert_eq!(
            *log.lock().expect("log"),
            vec![1, 2, 3],
            "every hook, including both inner-stack hooks, must run"
        );
        match flow {
            Flow::PatchRequest { patch } => {
                assert_eq!(patch.temperature, Some(0.2));
                assert_eq!(patch.max_tokens, Some(128));
                assert_eq!(patch.preamble.as_deref(), Some("outer"));
            }
            other => panic!("expected a merged PatchRequest, got {other:?}"),
        }
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
            stack.on_event(&ctx(), tool_call_event()).await,
            Flow::Continue
        ));
    }

    #[test]
    fn unit_hook_observes_no_event_kind() {
        // `impl AgentHook for ()` is the no-op hook: it must report interest in
        // *no* event kind, so the runner can skip building and dispatching even
        // the high-frequency delta events for it. The trait-default `observes`
        // returns `true`; `()` deliberately overrides it to `false`, so this is
        // the regression guard that the override stays in place.
        let all_kinds = [
            StepEventKind::CompletionCall,
            StepEventKind::CompletionResponse,
            StepEventKind::ModelTurnFinished,
            StepEventKind::InvalidToolCall,
            StepEventKind::ToolCall,
            StepEventKind::ToolResult,
            StepEventKind::TextDelta,
            StepEventKind::ToolCallDelta,
            StepEventKind::StreamResponseFinish,
        ];
        let unit_stack = HookStack::<M>::with(());
        for kind in all_kinds {
            assert!(
                !<() as AgentHook<M>>::observes(&(), kind),
                "the `()` no-op hook must not observe {kind:?}"
            );
            // A stack wrapping only `()` inherits that: it observes nothing, so
            // the runner skips delta dispatch for it too.
            assert!(
                !unit_stack.observes(kind),
                "a HookStack::with(()) must not observe {kind:?} either"
            );
        }
    }

    // --- RequestPatch merge unit tests ---

    #[test]
    fn merge_appends_extra_context_in_order() {
        let doc = |id: &str| crate::completion::Document {
            id: id.to_string(),
            text: String::new(),
            additional_props: Default::default(),
        };
        let a = RequestPatch::new().context(doc("a"));
        let b = RequestPatch::new().context(doc("b"));
        let merged = a.merge(b);
        let ids: Vec<&str> = merged.extra_context.iter().map(|d| d.id.as_str()).collect();
        assert_eq!(ids, vec!["a", "b"]);
    }

    #[test]
    fn merge_shallow_merges_additional_params_later_wins() {
        let a = RequestPatch::new().additional_params(serde_json::json!({"x": 1, "y": 2}));
        let b = RequestPatch::new().additional_params(serde_json::json!({"y": 3, "z": 4}));
        let merged = a.merge(b);
        assert_eq!(
            merged.additional_params,
            Some(serde_json::json!({"x": 1, "y": 3, "z": 4}))
        );
    }

    #[test]
    fn merge_scalar_last_writer_wins() {
        let a = RequestPatch::new().temperature(0.1);
        let b = RequestPatch::new().temperature(0.9);
        assert_eq!(a.merge(b).temperature, Some(0.9));
    }

    #[test]
    fn merge_active_tools_intersects() {
        let a = RequestPatch::new().active_tools(["search", "add", "sub"]);
        let b = RequestPatch::new().active_tools(["add", "sub", "mul"]);
        let merged = a.merge(b);
        assert_eq!(
            merged.active_tools,
            Some(vec!["add".to_string(), "sub".to_string()])
        );
    }

    #[test]
    fn merge_active_tools_empty_intersection_yields_empty() {
        let a = RequestPatch::new().active_tools(["search"]);
        let b = RequestPatch::new().active_tools(["add"]);
        let merged = a.merge(b);
        assert_eq!(merged.active_tools, Some(vec![]));
    }

    #[test]
    fn merge_one_sided_active_tools_keeps_the_present_list() {
        let a = RequestPatch::new().active_tools(["search"]);
        let b = RequestPatch::new();
        assert_eq!(a.merge(b).active_tools, Some(vec!["search".to_string()]));
    }

    // --- Scratchpad tests ---

    #[test]
    fn scratchpad_insert_get_update() {
        #[derive(Clone, Default, Debug, PartialEq)]
        struct Count(u32);

        let pad = Scratchpad::default();
        assert_eq!(pad.get::<Count>(), None);
        pad.update(|c: &mut Count| c.0 += 1);
        pad.update(|c: &mut Count| c.0 += 1);
        assert_eq!(pad.get::<Count>(), Some(Count(2)));
        assert!(pad.contains::<Count>());
        assert_eq!(pad.remove::<Count>(), Some(Count(2)));
        assert!(!pad.contains::<Count>());
    }

    #[test]
    fn scratchpad_is_shared_across_clones() {
        let pad = Scratchpad::default();
        let clone = pad.clone();
        pad.insert(7u32);
        // The clone shares the same underlying storage.
        assert_eq!(clone.get::<u32>(), Some(7));
    }

    #[test]
    fn hook_context_reports_identity_and_turn() {
        let ctx = HookContext::new(true, Some("agent".to_string()));
        assert!(ctx.is_streaming());
        assert_eq!(ctx.agent_name(), Some("agent"));
        assert_eq!(ctx.turn(), 0);
        ctx.set_turn(3);
        assert_eq!(ctx.turn(), 3);
        assert!(!ctx.run_id().as_str().is_empty());
    }

    /// Nested `HookStack` composition of the `ToolCall` chain: a rewrite inside an
    /// inner stack must survive a later short-circuit even though the inner stack
    /// is dispatched as a single hook. Regression coverage for the bug where
    /// `resolve_tool_call` consulted members via `on_event` (one `Flow`, so an
    /// inner rewrite was dropped behind an inner `Skip`/`Terminate`).
    mod nested_tool_call_resolution {
        use super::super::{AgentHook, Flow, HookContext, HookStack, StepEvent};
        use super::{M, ctx};
        use serde_json::{Value, json};

        /// Rewrites the tool args to a fixed value on `ToolCall`.
        struct RewriteHook(Value);
        impl AgentHook<M> for RewriteHook {
            async fn on_event(&self, _ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
                if let StepEvent::ToolCall { .. } = event {
                    Flow::rewrite_args(self.0.clone())
                } else {
                    Flow::cont()
                }
            }
        }

        /// Skips on `ToolCall`.
        struct SkipHook;
        impl AgentHook<M> for SkipHook {
            async fn on_event(&self, _ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
                if let StepEvent::ToolCall { .. } = event {
                    Flow::skip("denied")
                } else {
                    Flow::cont()
                }
            }
        }

        /// Terminates on `ToolCall`.
        struct TerminateHook;
        impl AgentHook<M> for TerminateHook {
            async fn on_event(&self, _ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
                if let StepEvent::ToolCall { .. } = event {
                    Flow::terminate("stop")
                } else {
                    Flow::cont()
                }
            }
        }

        /// Returns `Flow::Fail` on `ToolCall` — not honored there, so it is
        /// fail-closed by `run_single_tool`; `resolve_tool_call` returns it verbatim.
        struct FailHook;
        impl AgentHook<M> for FailHook {
            async fn on_event(&self, _ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
                if let StepEvent::ToolCall { .. } = event {
                    Flow::fail()
                } else {
                    Flow::cont()
                }
            }
        }

        /// Records the `args` each hook observes on `ToolCall`, to prove the
        /// rewritten args are threaded to hooks *after* the rewrite.
        #[derive(Clone, Default)]
        struct ArgsSpy(std::sync::Arc<std::sync::Mutex<Vec<String>>>);
        impl AgentHook<M> for ArgsSpy {
            async fn on_event(&self, _ctx: &HookContext, event: StepEvent<'_, M>) -> Flow {
                if let StepEvent::ToolCall { args, .. } = event {
                    self.0.lock().expect("spy").push(args.to_string());
                }
                Flow::cont()
            }
        }

        async fn resolve(stack: &HookStack<M>) -> (Flow, Option<Value>) {
            stack
                .resolve_tool_call(&ctx(), "add", Some("tc1"), "ic1", "{}")
                .await
        }

        #[tokio::test]
        async fn nested_rewrite_then_skip_preserves_rewrite() {
            // Inner stack: rewrite args, then skip. The rewrite must be salvaged.
            let mut inner = HookStack::<M>::new();
            inner.push(RewriteHook(json!({ "x": 41 })));
            inner.push(SkipHook);

            let mut outer = HookStack::<M>::new();
            outer.push(inner);

            let (flow, salvaged) = resolve(&outer).await;
            assert!(matches!(flow, Flow::Skip { .. }), "got {flow:?}");
            assert_eq!(
                salvaged,
                Some(json!({ "x": 41 })),
                "the inner rewrite must survive the inner skip through a nested stack"
            );
        }

        #[tokio::test]
        async fn nested_rewrite_then_terminate_preserves_rewrite() {
            let mut inner = HookStack::<M>::new();
            inner.push(RewriteHook(json!({ "x": 7 })));
            inner.push(TerminateHook);
            let mut outer = HookStack::<M>::new();
            outer.push(inner);

            let (flow, salvaged) = resolve(&outer).await;
            assert!(matches!(flow, Flow::Terminate { .. }), "got {flow:?}");
            assert_eq!(salvaged, Some(json!({ "x": 7 })));
        }

        #[tokio::test]
        async fn nested_rewrite_then_fail_closed_preserves_rewrite() {
            let mut inner = HookStack::<M>::new();
            inner.push(RewriteHook(json!({ "x": 9 })));
            inner.push(FailHook);
            let mut outer = HookStack::<M>::new();
            outer.push(inner);

            let (flow, salvaged) = resolve(&outer).await;
            // `Fail` is not honored for a tool call, but resolution returns it
            // verbatim (run_single_tool fail-closes it); the rewrite still survives.
            assert!(matches!(flow, Flow::Fail), "got {flow:?}");
            assert_eq!(salvaged, Some(json!({ "x": 9 })));
        }

        #[tokio::test]
        async fn outer_rewrite_then_nested_skip_preserves_outer_rewrite() {
            // Outer rewrite, then a nested stack that skips (without its own
            // rewrite). The outer rewrite must be salvaged, and the nested stack
            // must observe the outer-rewritten args.
            let spy = ArgsSpy::default();
            let mut inner = HookStack::<M>::new();
            inner.push(spy.clone());
            inner.push(SkipHook);

            let mut outer = HookStack::<M>::new();
            outer.push(RewriteHook(json!({ "x": 1, "y": 2 })));
            outer.push(inner);

            let (flow, salvaged) = resolve(&outer).await;
            assert!(matches!(flow, Flow::Skip { .. }), "got {flow:?}");
            assert_eq!(salvaged, Some(json!({ "x": 1, "y": 2 })));
            // The nested stack saw the outer-rewritten args, not the original `{}`.
            assert_eq!(
                spy.0.lock().expect("spy").as_slice(),
                [serde_json::to_string(&json!({ "x": 1, "y": 2 })).unwrap()],
            );
        }

        #[tokio::test]
        async fn deeply_nested_rewrite_then_skip_preserves_rewrite() {
            // Three levels: level3 rewrites+skips, wrapped twice.
            let mut level3 = HookStack::<M>::new();
            level3.push(RewriteHook(json!({ "deep": true })));
            level3.push(SkipHook);
            let mut level2 = HookStack::<M>::new();
            level2.push(level3);
            let mut level1 = HookStack::<M>::new();
            level1.push(level2);

            let (flow, salvaged) = resolve(&level1).await;
            assert!(matches!(flow, Flow::Skip { .. }), "got {flow:?}");
            assert_eq!(salvaged, Some(json!({ "deep": true })));
        }

        #[tokio::test]
        async fn nested_proceeding_rewrite_surfaces_as_rewrite_args() {
            // An inner stack that only rewrites (no short-circuit) surfaces the
            // rewrite through the flow itself, with no salvaged second element.
            let mut inner = HookStack::<M>::new();
            inner.push(RewriteHook(json!({ "x": 5 })));
            let mut outer = HookStack::<M>::new();
            outer.push(inner);

            let (flow, salvaged) = resolve(&outer).await;
            assert_eq!(
                flow,
                Flow::RewriteArgs {
                    args: json!({ "x": 5 })
                }
            );
            assert_eq!(salvaged, None);
        }
    }
}
