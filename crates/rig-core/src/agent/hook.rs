//! Type-safe lifecycle hooks for observing and steering agent runs.
//!
//! Each [`AgentHook`] lifecycle method receives one event-specific payload and
//! returns its matching action type, so invalid event/action combinations are
//! not representable:
//!
//! | Lifecycle | Action |
//! |---|---|
//! | [`AgentHook::on_completion_call`] | [`CompletionAction`] (continue, patch, stop) |
//! | completion response, model-turn finish, streamed deltas/finish | [`ObserveAction`] (continue, stop) |
//! | [`AgentHook::on_tool_call`] | [`ToolCallAction`] (run, rewrite, skip, stop) |
//! | [`AgentHook::on_tool_result`] | [`ToolResultAction`] (keep, rewrite, stop) |
//! | [`AgentHook::on_invalid_tool_call`] | [`InvalidToolCallAction`] (continue/fail, retry, repair, skip, stop) |
//!
//! Every method has a non-steering default; if every invalid-tool hook returns
//! `Continue`, Rig preserves its fail-fast behavior.
//!
//! # Composition
//!
//! [`HookStack`] invokes hooks in registration order. Completion request patches
//! accumulate and merge according to [`RequestPatch`]. Tool argument and result
//! rewrites chain, so each later hook sees the prior hook's presentation; raw
//! execution state and result metadata never change when a result is rewritten.
//! Skip and stop actions are terminal. Observe-only and invalid-tool lifecycles
//! use the first non-default action. Nested stacks preserve argument rewrites
//! even when a later inner hook skips or stops the call.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::{
    OneOrMany,
    completion::{CompletionModel, Document, Usage},
    json_utils,
    message::{AssistantContent, Message, ToolChoice},
    tool::{ToolContext, ToolExecutionView},
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
/// tool lifecycle
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
    // Reuses the tested `ToolContext` type-map as the storage, wrapped in
    // a shared lock so `&HookContext` hooks can mutate it. Under
    // `tool_concurrency > 1` several tools' `ToolCall`/`ToolResult` hooks may
    // touch this concurrently, so the lock is load-bearing, not decorative.
    inner: Arc<std::sync::Mutex<ToolContext>>,
}

impl Scratchpad {
    fn lock(&self) -> std::sync::MutexGuard<'_, ToolContext> {
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
}

impl std::fmt::Debug for Scratchpad {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scratchpad")
            .field("entries", &self.lock().len())
            .finish()
    }
}

/// Run-scoped context passed by shared reference to every event-specific hook
/// lifecycle method, such as [`AgentHook::on_completion_call`],
/// [`AgentHook::on_tool_call`], and [`AgentHook::on_tool_result`].
///
/// Carries the run's identity and a shared [`Scratchpad`]. It is a *driver*
/// construct built once per run by [`AgentRunner`](crate::agent::AgentRunner);
/// nothing here reaches the sans-IO [`AgentRun`](crate::agent::run::AgentRun)
/// state machine. Hooks hold it by `&`, so all fields are read via accessors and
/// run-scoped mutation goes through [`scratchpad`](Self::scratchpad).
///
/// One `HookContext` is shared by every hook invocation in a run. At
/// [`tool_concurrency`](crate::agent::AgentRunner::tool_concurrency)` > 1` the
/// tool lifecycle
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
    pub(crate) fn new(is_streaming: bool, agent_name: Option<String>) -> Self {
        Self {
            run_id: RunId::generate(),
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

/// Context passed to a hook on a [`InvalidToolCallContext`] event when the
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

/// Before a completion request is sent.
#[derive(Debug, Clone, Copy)]
pub struct CompletionCallEvent<'a> {
    /// Prompt message for this model turn.
    pub prompt: &'a Message,
    /// Chat history preceding the prompt.
    pub history: &'a [Message],
    /// One-based model-call index within the run.
    pub turn: usize,
}

/// After a non-streaming completion response is received.
pub struct CompletionResponseEvent<'a, M: CompletionModel> {
    /// Prompt message sent for this turn.
    pub prompt: &'a Message,
    /// Raw non-streaming completion response returned by the provider.
    pub response: &'a crate::completion::CompletionResponse<M::Response>,
}
impl<M: CompletionModel> Copy for CompletionResponseEvent<'_, M> {}
impl<M: CompletionModel> Clone for CompletionResponseEvent<'_, M> {
    fn clone(&self) -> Self {
        *self
    }
}

/// After a model turn is committed on either execution surface.
#[derive(Debug, Clone, Copy)]
pub struct ModelTurnFinishedEvent<'a> {
    /// One-based model-call index within the run.
    pub turn: usize,
    /// Canonical assistant content committed for the turn.
    pub content: &'a OneOrMany<AssistantContent>,
    /// Token usage reported for the turn.
    pub usage: Usage,
}

/// Before a valid tool call is executed.
#[derive(Debug, Clone, Copy)]
pub struct ToolCallEvent<'a> {
    /// Registered name of the tool about to run.
    pub tool_name: &'a str,
    /// Provider-supplied tool-call identifier, when available.
    pub tool_call_id: Option<&'a str>,
    /// Rig-generated identifier correlating call and result events.
    pub internal_call_id: &'a str,
    /// JSON arguments visible to this hook, including prior rewrites.
    pub args: &'a str,
}

/// After a tool call resolves or is skipped by policy.
#[derive(Debug, Clone, Copy)]
pub struct ToolResultEvent<'a> {
    /// Registered name of the resolved tool.
    pub tool_name: &'a str,
    /// Provider-supplied tool-call identifier, when available.
    pub tool_call_id: Option<&'a str>,
    /// Rig-generated identifier correlating call and result events.
    pub internal_call_id: &'a str,
    /// Effective JSON arguments used for execution or policy skip.
    pub args: &'a str,
    /// Presentation visible to this hook, including prior rewrites.
    pub result: &'a str,
    /// Raw execution state, unaffected by presentation rewrites.
    pub execution: ToolExecutionView<'a>,
    /// Per-call context carrying tool-authored result metadata.
    pub context: &'a ToolContext,
}

/// A streamed text delta.
#[derive(Debug, Clone, Copy)]
pub struct TextDeltaEvent<'a> {
    /// Newly received text fragment.
    pub delta: &'a str,
    /// Full text accumulated for the current turn so far.
    pub aggregated: &'a str,
}

/// A streamed tool-call argument delta.
#[derive(Debug, Clone, Copy)]
pub struct ToolCallDeltaEvent<'a> {
    /// Provider-supplied tool-call identifier.
    pub tool_call_id: &'a str,
    /// Rig-generated identifier correlating streamed call fragments.
    pub internal_call_id: &'a str,
    /// Tool name when supplied by this fragment.
    pub tool_name: Option<&'a str>,
    /// Newly received tool-argument fragment.
    pub delta: &'a str,
}

/// After a provider finishes a streaming text response.
pub struct StreamResponseFinishEvent<'a, M: CompletionModel> {
    /// Prompt message sent for this turn.
    pub prompt: &'a Message,
    /// Provider's final streaming response payload.
    pub response: &'a M::StreamingResponse,
}
impl<M: CompletionModel> Copy for StreamResponseFinishEvent<'_, M> {}
impl<M: CompletionModel> Clone for StreamResponseFinishEvent<'_, M> {
    fn clone(&self) -> Self {
        *self
    }
}

/// Event category used only as a streaming-dispatch performance hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum StepEventKind {
    /// Completion request lifecycle.
    CompletionCall,
    /// Non-streaming completion response lifecycle.
    CompletionResponse,
    /// Committed model-turn lifecycle shared by both execution surfaces.
    ModelTurnFinished,
    /// Invalid model-emitted tool-call lifecycle.
    InvalidToolCall,
    /// Pre-execution tool-call lifecycle.
    ToolCall,
    /// Post-execution tool-result lifecycle.
    ToolResult,
    /// Streamed text-delta lifecycle.
    TextDelta,
    /// Streamed tool-call-delta lifecycle.
    ToolCallDelta,
    /// Final streaming response lifecycle.
    StreamResponseFinish,
}

/// Partial, per-turn override for a completion request.
///
/// A [`HookStack`] merges patches in registration order: context appends,
/// provider parameters shallow-merge, active-tool allow-lists intersect, and
/// scalar/history values use the later writer. Patches never mutate the agent's
/// baseline configuration.
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
    /// An empty patch — a no-op, identical to returning the default action.
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

/// Action returned before a completion request.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum CompletionAction {
    /// Send the completion request unchanged.
    Continue,
    /// Merge the provided per-turn patch with patches from other hooks.
    Patch(RequestPatch),
    /// Stop the agent run with the supplied reason.
    Stop(String),
}
impl CompletionAction {
    /// Continue without changing the request.
    pub fn continue_run() -> Self {
        Self::Continue
    }

    /// Contribute a per-turn request patch.
    pub fn patch(patch: RequestPatch) -> Self {
        Self::Patch(patch)
    }

    /// Stop the run with a reason surfaced to the caller.
    pub fn stop(reason: impl Into<String>) -> Self {
        Self::Stop(reason.into())
    }
}

/// Action returned by observe-only lifecycle methods.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ObserveAction {
    /// Continue the run after observation.
    Continue,
    /// Stop the agent run with the supplied reason.
    Stop(String),
}
impl ObserveAction {
    /// Continue the run after observation.
    pub fn continue_run() -> Self {
        Self::Continue
    }

    /// Stop the run with a reason surfaced to the caller.
    pub fn stop(reason: impl Into<String>) -> Self {
        Self::Stop(reason.into())
    }
}

/// Action returned before tool execution.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ToolCallAction {
    /// Execute the tool with the currently effective arguments.
    Run,
    /// Replace the effective arguments before execution.
    Rewrite(serde_json::Value),
    /// Do not execute; return the supplied model-visible reason instead.
    Skip(String),
    /// Stop the entire agent run with the supplied reason.
    Stop(String),
}
impl ToolCallAction {
    /// Execute the tool with the currently effective arguments.
    pub fn run() -> Self {
        Self::Run
    }

    /// Replace the JSON arguments before executing the tool.
    pub fn rewrite(args: impl Into<serde_json::Value>) -> Self {
        Self::Rewrite(args.into())
    }

    /// Serialize a typed value into replacement JSON arguments.
    pub fn try_rewrite<T: serde::Serialize>(value: &T) -> Result<Self, serde_json::Error> {
        Ok(Self::Rewrite(serde_json::to_value(value)?))
    }

    /// Skip execution and return a model-visible reason.
    pub fn skip(reason: impl Into<String>) -> Self {
        Self::Skip(reason.into())
    }

    /// Stop the run before executing the tool.
    pub fn stop(reason: impl Into<String>) -> Self {
        Self::Stop(reason.into())
    }
}

/// Action returned after tool execution.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ToolResultAction {
    /// Keep the currently model-visible result unchanged.
    Keep,
    /// Replace only the result presented to later hooks and the model.
    Rewrite(String),
    /// Stop the entire agent run with the supplied reason.
    Stop(String),
}
impl ToolResultAction {
    /// Keep the currently model-visible result unchanged.
    pub fn keep() -> Self {
        Self::Keep
    }

    /// Replace the result presentation without changing raw execution data.
    pub fn rewrite(result: impl Into<String>) -> Self {
        Self::Rewrite(result.into())
    }

    /// Stop the run before presenting the result to the model.
    pub fn stop(reason: impl Into<String>) -> Self {
        Self::Stop(reason.into())
    }
}

/// Action returned for an invalid model-emitted tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum InvalidToolCallAction {
    /// Defer to a later hook; if every hook defers, preserve fail-fast behavior.
    Continue,
    /// Reject the invalid call and fail the run immediately.
    Fail,
    /// Retry the model turn with corrective feedback.
    Retry(String),
    /// Replace the emitted tool name and revalidate it before execution.
    Repair(String),
    /// Record a synthetic model-visible result without executing a tool.
    Skip(String),
    /// Stop the entire run with the supplied reason.
    Stop(String),
}
impl InvalidToolCallAction {
    /// Defer to a later hook.
    ///
    /// If every hook defers, Rig rejects the invalid call and fails fast.
    pub fn continue_run() -> Self {
        Self::Continue
    }

    /// Reject the invalid call and fail the run immediately.
    pub fn fail() -> Self {
        Self::Fail
    }

    /// Retry the model turn with corrective feedback.
    pub fn retry(feedback: impl Into<String>) -> Self {
        Self::Retry(feedback.into())
    }

    /// Repair the emitted tool name and request revalidation.
    pub fn repair(tool_name: impl Into<String>) -> Self {
        Self::Repair(tool_name.into())
    }

    /// Skip execution and record a synthetic model-visible result.
    pub fn skip(reason: impl Into<String>) -> Self {
        Self::Skip(reason.into())
    }

    /// Stop the run with a reason surfaced to the caller.
    pub fn stop(reason: impl Into<String>) -> Self {
        Self::Stop(reason.into())
    }
}

/// Per-run lifecycle observer and steerer with event-specific actions.
pub trait AgentHook<M>: WasmCompatSend + WasmCompatSync
where
    M: CompletionModel,
{
    /// Observe or steer a completion request before it is sent.
    ///
    /// The default continues unchanged. In a [`HookStack`], every patch is
    /// merged in registration order; only a stop action short-circuits.
    fn on_completion_call(
        &self,
        ctx: &HookContext,
        event: CompletionCallEvent<'_>,
    ) -> impl std::future::Future<Output = CompletionAction> + WasmCompatSend {
        let _ = (ctx, event);
        async { CompletionAction::Continue }
    }
    /// Observe a raw non-streaming completion response.
    ///
    /// The default continues. A stack returns the first stop action.
    fn on_completion_response(
        &self,
        ctx: &HookContext,
        event: CompletionResponseEvent<'_, M>,
    ) -> impl std::future::Future<Output = ObserveAction> + WasmCompatSend {
        let _ = (ctx, event);
        async { ObserveAction::Continue }
    }
    /// Observe a committed model turn on either execution surface.
    ///
    /// The default continues. A stack returns the first stop action.
    fn on_model_turn_finished(
        &self,
        ctx: &HookContext,
        event: ModelTurnFinishedEvent<'_>,
    ) -> impl std::future::Future<Output = ObserveAction> + WasmCompatSend {
        let _ = (ctx, event);
        async { ObserveAction::Continue }
    }
    /// Choose recovery for an unknown or disallowed model-emitted tool call.
    ///
    /// The default defers to later hooks; if every hook defers, Rig fails fast.
    fn on_invalid_tool_call(
        &self,
        ctx: &HookContext,
        event: &InvalidToolCallContext,
    ) -> impl std::future::Future<Output = InvalidToolCallAction> + WasmCompatSend {
        let _ = (ctx, event);
        async { InvalidToolCallAction::Continue }
    }
    /// Observe or steer a valid tool call before execution.
    ///
    /// The default runs the tool. In a stack, rewrites chain in registration
    /// order and each later hook sees the effective rewritten arguments.
    fn on_tool_call(
        &self,
        ctx: &HookContext,
        event: ToolCallEvent<'_>,
    ) -> impl std::future::Future<Output = ToolCallAction> + WasmCompatSend {
        let _ = (ctx, event);
        async { ToolCallAction::Run }
    }
    /// Observe or rewrite a tool result before model presentation.
    ///
    /// The default keeps the result. In a stack, presentation rewrites chain;
    /// raw execution state and result metadata remain unchanged.
    fn on_tool_result(
        &self,
        ctx: &HookContext,
        event: ToolResultEvent<'_>,
    ) -> impl std::future::Future<Output = ToolResultAction> + WasmCompatSend {
        let _ = (ctx, event);
        async { ToolResultAction::Keep }
    }
    /// Observe a streamed text delta.
    ///
    /// The default continues. A stack returns the first stop action.
    fn on_text_delta(
        &self,
        ctx: &HookContext,
        event: TextDeltaEvent<'_>,
    ) -> impl std::future::Future<Output = ObserveAction> + WasmCompatSend {
        let _ = (ctx, event);
        async { ObserveAction::Continue }
    }
    /// Observe a streamed tool-call argument delta.
    ///
    /// The default continues. A stack returns the first stop action.
    fn on_tool_call_delta(
        &self,
        ctx: &HookContext,
        event: ToolCallDeltaEvent<'_>,
    ) -> impl std::future::Future<Output = ObserveAction> + WasmCompatSend {
        let _ = (ctx, event);
        async { ObserveAction::Continue }
    }
    /// Observe a provider's final streaming response.
    ///
    /// The default continues. A stack returns the first stop action.
    fn on_stream_response_finish(
        &self,
        ctx: &HookContext,
        event: StreamResponseFinishEvent<'_, M>,
    ) -> impl std::future::Future<Output = ObserveAction> + WasmCompatSend {
        let _ = (ctx, event);
        async { ObserveAction::Continue }
    }

    #[doc(hidden)]
    fn resolve_tool_call<'a>(
        &'a self,
        ctx: &'a HookContext,
        event: ToolCallEvent<'a>,
    ) -> impl std::future::Future<Output = (ToolCallAction, Option<serde_json::Value>)> + WasmCompatSend
    {
        async move { (self.on_tool_call(ctx, event).await, None) }
    }

    /// Report interest in an event category.
    ///
    /// This is a performance hint used to avoid constructing high-frequency
    /// streaming delta events when no registered hook observes them. The
    /// default observes every category.
    fn observes(&self, kind: StepEventKind) -> bool {
        let _ = kind;
        true
    }
}

impl<M: CompletionModel> AgentHook<M> for () {
    fn observes(&self, _kind: StepEventKind) -> bool {
        false
    }
}

trait DynAgentHook<M>: WasmCompatSend + WasmCompatSync
where
    M: CompletionModel,
{
    fn completion_call<'a>(
        &'a self,
        c: &'a HookContext,
        e: CompletionCallEvent<'a>,
    ) -> WasmBoxedFuture<'a, CompletionAction>
    where
        M: 'a;
    fn completion_response<'a>(
        &'a self,
        c: &'a HookContext,
        e: CompletionResponseEvent<'a, M>,
    ) -> WasmBoxedFuture<'a, ObserveAction>
    where
        M: 'a;
    fn model_turn_finished<'a>(
        &'a self,
        c: &'a HookContext,
        e: ModelTurnFinishedEvent<'a>,
    ) -> WasmBoxedFuture<'a, ObserveAction>
    where
        M: 'a;
    fn invalid_tool_call<'a>(
        &'a self,
        c: &'a HookContext,
        e: &'a InvalidToolCallContext,
    ) -> WasmBoxedFuture<'a, InvalidToolCallAction>
    where
        M: 'a;
    fn resolve_tool_call<'a>(
        &'a self,
        c: &'a HookContext,
        e: ToolCallEvent<'a>,
    ) -> WasmBoxedFuture<'a, (ToolCallAction, Option<serde_json::Value>)>
    where
        M: 'a;
    fn tool_result<'a>(
        &'a self,
        c: &'a HookContext,
        e: ToolResultEvent<'a>,
    ) -> WasmBoxedFuture<'a, ToolResultAction>
    where
        M: 'a;
    fn text_delta<'a>(
        &'a self,
        c: &'a HookContext,
        e: TextDeltaEvent<'a>,
    ) -> WasmBoxedFuture<'a, ObserveAction>
    where
        M: 'a;
    fn tool_call_delta<'a>(
        &'a self,
        c: &'a HookContext,
        e: ToolCallDeltaEvent<'a>,
    ) -> WasmBoxedFuture<'a, ObserveAction>
    where
        M: 'a;
    fn stream_response_finish<'a>(
        &'a self,
        c: &'a HookContext,
        e: StreamResponseFinishEvent<'a, M>,
    ) -> WasmBoxedFuture<'a, ObserveAction>
    where
        M: 'a;
    fn observes(&self, kind: StepEventKind) -> bool;
}
impl<M, H> DynAgentHook<M> for H
where
    M: CompletionModel,
    H: AgentHook<M>,
{
    fn completion_call<'a>(
        &'a self,
        c: &'a HookContext,
        e: CompletionCallEvent<'a>,
    ) -> WasmBoxedFuture<'a, CompletionAction>
    where
        M: 'a,
    {
        Box::pin(self.on_completion_call(c, e))
    }
    fn completion_response<'a>(
        &'a self,
        c: &'a HookContext,
        e: CompletionResponseEvent<'a, M>,
    ) -> WasmBoxedFuture<'a, ObserveAction>
    where
        M: 'a,
    {
        Box::pin(self.on_completion_response(c, e))
    }
    fn model_turn_finished<'a>(
        &'a self,
        c: &'a HookContext,
        e: ModelTurnFinishedEvent<'a>,
    ) -> WasmBoxedFuture<'a, ObserveAction>
    where
        M: 'a,
    {
        Box::pin(self.on_model_turn_finished(c, e))
    }
    fn invalid_tool_call<'a>(
        &'a self,
        c: &'a HookContext,
        e: &'a InvalidToolCallContext,
    ) -> WasmBoxedFuture<'a, InvalidToolCallAction>
    where
        M: 'a,
    {
        Box::pin(self.on_invalid_tool_call(c, e))
    }
    fn resolve_tool_call<'a>(
        &'a self,
        c: &'a HookContext,
        e: ToolCallEvent<'a>,
    ) -> WasmBoxedFuture<'a, (ToolCallAction, Option<serde_json::Value>)>
    where
        M: 'a,
    {
        Box::pin(AgentHook::resolve_tool_call(self, c, e))
    }
    fn tool_result<'a>(
        &'a self,
        c: &'a HookContext,
        e: ToolResultEvent<'a>,
    ) -> WasmBoxedFuture<'a, ToolResultAction>
    where
        M: 'a,
    {
        Box::pin(self.on_tool_result(c, e))
    }
    fn text_delta<'a>(
        &'a self,
        c: &'a HookContext,
        e: TextDeltaEvent<'a>,
    ) -> WasmBoxedFuture<'a, ObserveAction>
    where
        M: 'a,
    {
        Box::pin(self.on_text_delta(c, e))
    }
    fn tool_call_delta<'a>(
        &'a self,
        c: &'a HookContext,
        e: ToolCallDeltaEvent<'a>,
    ) -> WasmBoxedFuture<'a, ObserveAction>
    where
        M: 'a,
    {
        Box::pin(self.on_tool_call_delta(c, e))
    }
    fn stream_response_finish<'a>(
        &'a self,
        c: &'a HookContext,
        e: StreamResponseFinishEvent<'a, M>,
    ) -> WasmBoxedFuture<'a, ObserveAction>
    where
        M: 'a,
    {
        Box::pin(self.on_stream_response_finish(c, e))
    }
    fn observes(&self, k: StepEventKind) -> bool {
        AgentHook::observes(self, k)
    }
}

/// Ordered composition of lifecycle hooks.
///
/// Completion patches from every hook merge in registration order. Tool-call
/// and tool-result rewrites chain so later hooks see the current presentation.
/// Skip and stop actions terminate those chains. Observe-only and invalid-call
/// lifecycles return the first non-default action. Nested stacks preserve the
/// same behavior, including salvaging an argument rewrite before a later skip
/// or stop.
pub struct HookStack<M: CompletionModel> {
    hooks: Vec<Arc<dyn DynAgentHook<M>>>,
}
impl<M: CompletionModel> Clone for HookStack<M> {
    fn clone(&self) -> Self {
        Self {
            hooks: self.hooks.clone(),
        }
    }
}
impl<M: CompletionModel> Default for HookStack<M> {
    fn default() -> Self {
        Self { hooks: Vec::new() }
    }
}
impl<M: CompletionModel> std::fmt::Debug for HookStack<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HookStack")
            .field("len", &self.hooks.len())
            .finish()
    }
}
impl<M: CompletionModel> HookStack<M> {
    /// Create an empty stack that returns each lifecycle's default action.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a stack containing one hook.
    pub fn with<H: AgentHook<M> + 'static>(hook: H) -> Self {
        let mut s = Self::new();
        s.push(hook);
        s
    }

    /// Append a hook to the end of the registration order.
    pub fn push<H: AgentHook<M> + 'static>(&mut self, hook: H) {
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

impl<M: CompletionModel> AgentHook<M> for HookStack<M> {
    async fn on_completion_call(
        &self,
        ctx: &HookContext,
        event: CompletionCallEvent<'_>,
    ) -> CompletionAction {
        let mut merged: Option<RequestPatch> = None;
        for hook in &self.hooks {
            match hook.completion_call(ctx, event).await {
                CompletionAction::Continue => {}
                CompletionAction::Patch(patch) => {
                    merged = Some(match merged {
                        Some(accumulated) => accumulated.merge(patch),
                        None => patch,
                    });
                }
                stop @ CompletionAction::Stop(_) => return stop,
            }
        }
        match merged {
            Some(patch) if !patch.is_empty() => CompletionAction::Patch(patch),
            _ => CompletionAction::Continue,
        }
    }
    async fn on_completion_response(
        &self,
        ctx: &HookContext,
        event: CompletionResponseEvent<'_, M>,
    ) -> ObserveAction {
        for h in &self.hooks {
            let a = h.completion_response(ctx, event).await;
            if !matches!(a, ObserveAction::Continue) {
                return a;
            }
        }
        ObserveAction::Continue
    }
    async fn on_model_turn_finished(
        &self,
        ctx: &HookContext,
        event: ModelTurnFinishedEvent<'_>,
    ) -> ObserveAction {
        for h in &self.hooks {
            let a = h.model_turn_finished(ctx, event).await;
            if !matches!(a, ObserveAction::Continue) {
                return a;
            }
        }
        ObserveAction::Continue
    }
    async fn on_invalid_tool_call(
        &self,
        ctx: &HookContext,
        event: &InvalidToolCallContext,
    ) -> InvalidToolCallAction {
        for h in &self.hooks {
            let a = h.invalid_tool_call(ctx, event).await;
            if !matches!(a, InvalidToolCallAction::Continue) {
                return a;
            }
        }
        InvalidToolCallAction::Continue
    }
    async fn resolve_tool_call<'a>(
        &'a self,
        ctx: &'a HookContext,
        event: ToolCallEvent<'a>,
    ) -> (ToolCallAction, Option<serde_json::Value>) {
        let mut effective = None;
        for h in &self.hooks {
            let rendered = effective.as_ref().map(json_utils::value_to_json_string);
            let e = ToolCallEvent {
                args: rendered.as_deref().unwrap_or(event.args),
                ..event
            };
            let (action, salvaged) = h.resolve_tool_call(ctx, e).await;
            if let Some(v) = salvaged {
                effective = Some(v)
            }
            match action {
                ToolCallAction::Run => {}
                ToolCallAction::Rewrite(v) => effective = Some(v),
                other => return (other, effective),
            }
        }
        match effective {
            Some(v) => (ToolCallAction::Rewrite(v), None),
            None => (ToolCallAction::Run, None),
        }
    }
    async fn on_tool_call(&self, ctx: &HookContext, event: ToolCallEvent<'_>) -> ToolCallAction {
        AgentHook::resolve_tool_call(self, ctx, event).await.0
    }
    async fn on_tool_result(
        &self,
        ctx: &HookContext,
        event: ToolResultEvent<'_>,
    ) -> ToolResultAction {
        let mut effective = None;
        for h in &self.hooks {
            let e = ToolResultEvent {
                result: effective.as_deref().unwrap_or(event.result),
                ..event
            };
            match h.tool_result(ctx, e).await {
                ToolResultAction::Keep => {}
                ToolResultAction::Rewrite(v) => effective = Some(v),
                other => return other,
            }
        }
        match effective {
            Some(v) => ToolResultAction::Rewrite(v),
            None => ToolResultAction::Keep,
        }
    }
    async fn on_text_delta(&self, ctx: &HookContext, event: TextDeltaEvent<'_>) -> ObserveAction {
        for h in &self.hooks {
            let a = h.text_delta(ctx, event).await;
            if !matches!(a, ObserveAction::Continue) {
                return a;
            }
        }
        ObserveAction::Continue
    }
    async fn on_tool_call_delta(
        &self,
        ctx: &HookContext,
        event: ToolCallDeltaEvent<'_>,
    ) -> ObserveAction {
        for h in &self.hooks {
            let a = h.tool_call_delta(ctx, event).await;
            if !matches!(a, ObserveAction::Continue) {
                return a;
            }
        }
        ObserveAction::Continue
    }
    async fn on_stream_response_finish(
        &self,
        ctx: &HookContext,
        event: StreamResponseFinishEvent<'_, M>,
    ) -> ObserveAction {
        for h in &self.hooks {
            let a = h.stream_response_finish(ctx, event).await;
            if !matches!(a, ObserveAction::Continue) {
                return a;
            }
        }
        ObserveAction::Continue
    }
    fn observes(&self, k: StepEventKind) -> bool {
        self.hooks.iter().any(|h| h.observes(k))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::{
        AgentHook, CompletionAction, CompletionCallEvent, HookContext, HookStack, ObserveAction,
        RequestPatch, Scratchpad, StepEventKind, TextDeltaEvent, ToolCallAction, ToolCallEvent,
        ToolResultAction, ToolResultEvent,
    };
    use crate::{
        test_utils::MockCompletionModel,
        tool::{ToolContext, ToolExecutionView},
    };

    type M = MockCompletionModel;

    fn ctx() -> HookContext {
        HookContext::new(false, Some("test-agent".to_string()))
    }

    /// Pushes its label for each lifecycle method used below and optionally stops.
    struct Recorder {
        label: u32,
        log: Arc<Mutex<Vec<u32>>>,
        stop: bool,
    }

    impl AgentHook<M> for Recorder {
        async fn on_completion_call(
            &self,
            _ctx: &HookContext,
            _event: CompletionCallEvent<'_>,
        ) -> CompletionAction {
            self.log.lock().expect("log").push(self.label);
            if self.stop {
                CompletionAction::stop("stop")
            } else {
                CompletionAction::continue_run()
            }
        }

        async fn on_tool_call(
            &self,
            _ctx: &HookContext,
            _event: ToolCallEvent<'_>,
        ) -> ToolCallAction {
            self.log.lock().expect("log").push(self.label);
            if self.stop {
                ToolCallAction::stop("stop")
            } else {
                ToolCallAction::run()
            }
        }

        async fn on_text_delta(
            &self,
            _ctx: &HookContext,
            _event: TextDeltaEvent<'_>,
        ) -> ObserveAction {
            self.log.lock().expect("log").push(self.label);
            if self.stop {
                ObserveAction::stop("stop")
            } else {
                ObserveAction::continue_run()
            }
        }
    }

    /// Observes exactly one event kind (used to probe stack-level `observes`).
    struct ObservesOnly(StepEventKind);

    impl AgentHook<M> for ObservesOnly {
        fn observes(&self, kind: StepEventKind) -> bool {
            kind == self.0
        }
    }

    /// Returns a fixed patch on completion calls and records its label.
    struct Patcher {
        label: u32,
        log: Arc<Mutex<Vec<u32>>>,
        patch: RequestPatch,
    }

    impl AgentHook<M> for Patcher {
        async fn on_completion_call(
            &self,
            _ctx: &HookContext,
            _event: CompletionCallEvent<'_>,
        ) -> CompletionAction {
            self.log.lock().expect("log").push(self.label);
            CompletionAction::patch(self.patch.clone())
        }
    }

    fn tool_call_event() -> ToolCallEvent<'static> {
        ToolCallEvent {
            tool_name: "add",
            tool_call_id: Some("tc1"),
            internal_call_id: "ic1",
            args: "{}",
        }
    }

    fn completion_call_event() -> CompletionCallEvent<'static> {
        static PROMPT: std::sync::OnceLock<crate::message::Message> = std::sync::OnceLock::new();
        let prompt = PROMPT.get_or_init(|| crate::message::Message::user("hi"));
        CompletionCallEvent {
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

        let action = stack.on_tool_call(&ctx(), tool_call_event()).await;

        assert!(matches!(action, ToolCallAction::Run));
        assert_eq!(*log.lock().expect("log"), vec![1, 2]);
    }

    #[tokio::test]
    async fn first_stop_short_circuits_on_chained_tool_call() {
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

        let action = stack.on_tool_call(&ctx(), tool_call_event()).await;

        assert!(matches!(action, ToolCallAction::Stop(_)));
        assert_eq!(
            *log.lock().expect("log"),
            vec![1],
            "a later hook must not run after an earlier hook stops"
        );
    }

    #[tokio::test]
    async fn first_stop_short_circuits_on_observe_only_events() {
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

        let action = stack
            .on_text_delta(
                &ctx(),
                TextDeltaEvent {
                    delta: "hi",
                    aggregated: "hi",
                },
            )
            .await;

        assert!(matches!(action, ObserveAction::Stop(_)));
        assert_eq!(
            *log.lock().expect("log"),
            vec![1],
            "a later hook must not run after an earlier hook stops an observe-only event"
        );
    }

    #[tokio::test]
    async fn completion_call_patches_accumulate_and_consult_every_hook() {
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

        let action = stack
            .on_completion_call(&ctx(), completion_call_event())
            .await;

        assert_eq!(
            *log.lock().expect("log"),
            vec![1, 2],
            "both hooks must run; a mergeable patch does not short-circuit"
        );
        match action {
            CompletionAction::Patch(patch) => {
                assert_eq!(patch.temperature, Some(0.1));
                assert_eq!(patch.max_tokens, Some(256));
            }
            other => panic!("expected a merged patch, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn completion_call_stop_short_circuits_and_discards_patch() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut stack = HookStack::<M>::with(Patcher {
            label: 1,
            log: log.clone(),
            patch: RequestPatch::new().temperature(0.1),
        });
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

        let action = stack
            .on_completion_call(&ctx(), completion_call_event())
            .await;

        assert!(matches!(action, CompletionAction::Stop(_)));
        assert_eq!(
            *log.lock().expect("log"),
            vec![1, 2],
            "hook 3 must not run after a stop"
        );
    }

    #[tokio::test]
    async fn nested_stack_composes_patches_without_inner_short_circuit() {
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

        let action = outer
            .on_completion_call(&ctx(), completion_call_event())
            .await;

        assert_eq!(
            *log.lock().expect("log"),
            vec![1, 2, 3],
            "every hook, including both inner-stack hooks, must run"
        );
        match action {
            CompletionAction::Patch(patch) => {
                assert_eq!(patch.temperature, Some(0.2));
                assert_eq!(patch.max_tokens, Some(128));
                assert_eq!(patch.preamble.as_deref(), Some("outer"));
            }
            other => panic!("expected a merged patch, got {other:?}"),
        }
    }

    #[test]
    fn stack_observes_is_the_or_of_its_members() {
        let mut stack = HookStack::<M>::with(ObservesOnly(StepEventKind::ToolCall));
        stack.push(ObservesOnly(StepEventKind::ToolResult));

        assert!(stack.observes(StepEventKind::ToolCall));
        assert!(stack.observes(StepEventKind::ToolResult));
        assert!(!stack.observes(StepEventKind::TextDelta));
    }

    #[tokio::test]
    async fn empty_stack_uses_default_actions_and_observes_nothing() {
        let stack = HookStack::<M>::new();

        assert!(stack.is_empty());
        assert!(!stack.observes(StepEventKind::ToolCall));
        assert!(!stack.observes(StepEventKind::TextDelta));
        assert!(matches!(
            stack.on_tool_call(&ctx(), tool_call_event()).await,
            ToolCallAction::Run
        ));
        assert!(matches!(
            stack
                .on_text_delta(
                    &ctx(),
                    TextDeltaEvent {
                        delta: "hi",
                        aggregated: "hi",
                    },
                )
                .await,
            ObserveAction::Continue
        ));
    }

    #[test]
    fn unit_hook_observes_no_event_kind() {
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
            assert!(!<() as AgentHook<M>>::observes(&(), kind));
            assert!(!unit_stack.observes(kind));
        }
    }

    #[test]
    fn merge_appends_extra_context_in_order() {
        let doc = |id: &str| crate::completion::Document {
            id: id.to_string(),
            text: String::new(),
            additional_props: Default::default(),
        };
        let merged = RequestPatch::new()
            .context(doc("a"))
            .merge(RequestPatch::new().context(doc("b")));
        let ids: Vec<&str> = merged.extra_context.iter().map(|d| d.id.as_str()).collect();
        assert_eq!(ids, vec!["a", "b"]);
    }

    #[test]
    fn merge_shallow_merges_additional_params_later_wins() {
        let merged = RequestPatch::new()
            .additional_params(serde_json::json!({"x": 1, "y": 2}))
            .merge(RequestPatch::new().additional_params(serde_json::json!({"y": 3, "z": 4})));
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
        assert_eq!(
            a.merge(b).active_tools,
            Some(vec!["add".to_string(), "sub".to_string()])
        );
    }

    #[test]
    fn merge_active_tools_empty_intersection_yields_empty() {
        let a = RequestPatch::new().active_tools(["search"]);
        let b = RequestPatch::new().active_tools(["add"]);
        assert_eq!(a.merge(b).active_tools, Some(vec![]));
    }

    #[test]
    fn merge_one_sided_active_tools_keeps_the_present_list() {
        let a = RequestPatch::new().active_tools(["search"]);
        assert_eq!(
            a.merge(RequestPatch::new()).active_tools,
            Some(vec!["search".to_string()])
        );
    }

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

    mod nested_tool_call_resolution {
        use super::super::{AgentHook, HookContext, HookStack, ToolCallAction, ToolCallEvent};
        use super::{M, ctx};
        use serde_json::{Value, json};

        struct RewriteHook(Value);
        impl AgentHook<M> for RewriteHook {
            async fn on_tool_call(
                &self,
                _ctx: &HookContext,
                _event: ToolCallEvent<'_>,
            ) -> ToolCallAction {
                ToolCallAction::rewrite(self.0.clone())
            }
        }

        struct SkipHook;
        impl AgentHook<M> for SkipHook {
            async fn on_tool_call(
                &self,
                _ctx: &HookContext,
                _event: ToolCallEvent<'_>,
            ) -> ToolCallAction {
                ToolCallAction::skip("denied")
            }
        }

        struct StopHook;
        impl AgentHook<M> for StopHook {
            async fn on_tool_call(
                &self,
                _ctx: &HookContext,
                _event: ToolCallEvent<'_>,
            ) -> ToolCallAction {
                ToolCallAction::stop("stop")
            }
        }

        #[derive(Clone, Default)]
        struct ArgsSpy(std::sync::Arc<std::sync::Mutex<Vec<String>>>);
        impl AgentHook<M> for ArgsSpy {
            async fn on_tool_call(
                &self,
                _ctx: &HookContext,
                event: ToolCallEvent<'_>,
            ) -> ToolCallAction {
                self.0.lock().expect("spy").push(event.args.to_string());
                ToolCallAction::run()
            }
        }

        async fn resolve(stack: &HookStack<M>) -> (ToolCallAction, Option<Value>) {
            AgentHook::resolve_tool_call(
                stack,
                &ctx(),
                ToolCallEvent {
                    tool_name: "add",
                    tool_call_id: Some("tc1"),
                    internal_call_id: "ic1",
                    args: "{}",
                },
            )
            .await
        }

        #[tokio::test]
        async fn nested_rewrite_then_skip_preserves_rewrite() {
            let mut inner = HookStack::<M>::new();
            inner.push(RewriteHook(json!({ "x": 41 })));
            inner.push(SkipHook);
            let outer = HookStack::<M>::with(inner);

            let (action, salvaged) = resolve(&outer).await;
            assert!(matches!(action, ToolCallAction::Skip(_)), "got {action:?}");
            assert_eq!(salvaged, Some(json!({ "x": 41 })));
        }

        #[tokio::test]
        async fn nested_rewrite_then_stop_preserves_rewrite() {
            let mut inner = HookStack::<M>::new();
            inner.push(RewriteHook(json!({ "x": 7 })));
            inner.push(StopHook);
            let outer = HookStack::<M>::with(inner);

            let (action, salvaged) = resolve(&outer).await;
            assert!(matches!(action, ToolCallAction::Stop(_)), "got {action:?}");
            assert_eq!(salvaged, Some(json!({ "x": 7 })));
        }

        #[tokio::test]
        async fn outer_rewrite_then_nested_skip_preserves_outer_rewrite() {
            let spy = ArgsSpy::default();
            let mut inner = HookStack::<M>::new();
            inner.push(spy.clone());
            inner.push(SkipHook);

            let mut outer = HookStack::<M>::new();
            outer.push(RewriteHook(json!({ "x": 1, "y": 2 })));
            outer.push(inner);

            let (action, salvaged) = resolve(&outer).await;
            assert!(matches!(action, ToolCallAction::Skip(_)), "got {action:?}");
            assert_eq!(salvaged, Some(json!({ "x": 1, "y": 2 })));
            assert_eq!(
                spy.0.lock().expect("spy").as_slice(),
                [serde_json::to_string(&json!({ "x": 1, "y": 2 })).unwrap()],
            );
        }

        #[tokio::test]
        async fn deeply_nested_rewrite_then_skip_preserves_rewrite() {
            let mut level3 = HookStack::<M>::new();
            level3.push(RewriteHook(json!({ "deep": true })));
            level3.push(SkipHook);
            let level2 = HookStack::<M>::with(level3);
            let level1 = HookStack::<M>::with(level2);

            let (action, salvaged) = resolve(&level1).await;
            assert!(matches!(action, ToolCallAction::Skip(_)), "got {action:?}");
            assert_eq!(salvaged, Some(json!({ "deep": true })));
        }

        #[tokio::test]
        async fn nested_proceeding_rewrite_surfaces_as_rewrite_action() {
            let inner = HookStack::<M>::with(RewriteHook(json!({ "x": 5 })));
            let outer = HookStack::<M>::with(inner);

            let (action, salvaged) = resolve(&outer).await;
            assert_eq!(action, ToolCallAction::Rewrite(json!({ "x": 5 })));
            assert_eq!(salvaged, None);
        }
    }

    struct ResultRewrite(&'static str, Arc<Mutex<Vec<String>>>);
    impl AgentHook<M> for ResultRewrite {
        async fn on_tool_result(
            &self,
            _ctx: &HookContext,
            event: ToolResultEvent<'_>,
        ) -> ToolResultAction {
            self.1.lock().expect("results").push(event.result.into());
            ToolResultAction::rewrite(self.0)
        }
    }

    #[tokio::test]
    async fn result_rewrites_chain_without_changing_raw_execution_view() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut stack = HookStack::<M>::new();
        stack.push(ResultRewrite("a", seen.clone()));
        stack.push(ResultRewrite("b", seen.clone()));
        let context = ToolContext::new();

        let action = stack
            .on_tool_result(
                &ctx(),
                ToolResultEvent {
                    tool_name: "t",
                    tool_call_id: None,
                    internal_call_id: "i",
                    args: "{}",
                    result: "raw",
                    execution: ToolExecutionView::Success,
                    context: &context,
                },
            )
            .await;

        assert_eq!(*seen.lock().expect("results"), vec!["raw", "a"]);
        assert_eq!(action, ToolResultAction::Rewrite("b".into()));
    }
}
