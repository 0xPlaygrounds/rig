//! Event-specific hooks for observing and steering an agent run.
//!
//! [`AgentHook`] replaces the old universal event/action pair with one lifecycle
//! method and one action type per event. Unsupported combinations are therefore
//! rejected by the compiler instead of being interpreted at runtime.
//!
//! Hooks run in registration order through [`HookStack`]. Completion-call
//! [`RequestPatch`] values accumulate and merge; tool-call argument rewrites and
//! tool-result presentation rewrites chain into later hooks. Nested stacks obey
//! the same rules as flat stacks, including preserving an argument rewrite when
//! an inner stack later skips or stops. Every stop action short-circuits the
//! remaining hooks for that event.
//!
//! Register observe-only hooks before steering hooks when every observation is
//! required: a steering stop intentionally prevents later observers from
//! running. Tool-result rewrites change the effective `presentation` sent to
//! the model and recorded as result-content telemetry. The
//! [`ToolResultEvent::raw_result`] and its [`ToolResultEvent::tool_context`]
//! remain unchanged for policy decisions and execution-outcome metadata. A
//! tool-result stop omits result content from telemetry.
//!
//! Blocking and streaming agents share the same request, tool-call, and
//! tool-result resolution path. Streaming adds delta-specific observations, but
//! shared lifecycle actions have identical semantics on both surfaces.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{future::Future, sync::Arc};

use crate::tool::extensions::TypeMap;
use crate::{
    OneOrMany,
    completion::{CompletionModel, Document, Usage},
    json_utils,
    message::{AssistantContent, Message, ToolChoice},
    tool::{ToolContext, ToolOutput, ToolResult},
    wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync},
};

/// Opaque stable identifier for one logical agent run, including durable
/// checkpoint resumption in another process.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RunId(String);

impl RunId {
    pub(crate) fn generate() -> Self {
        Self(crate::id::generate())
    }

    pub(crate) fn from_stored(value: String) -> Self {
        Self(value)
    }

    /// Identifier as text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for RunId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Run-scoped typed storage shared by hooks.
#[derive(Clone, Default)]
pub struct Scratchpad {
    inner: Arc<std::sync::Mutex<TypeMap>>,
}

impl Scratchpad {
    fn lock(&self) -> std::sync::MutexGuard<'_, TypeMap> {
        self.inner.lock().unwrap_or_else(|error| error.into_inner())
    }

    /// Insert a value.
    pub fn insert<T>(&self, value: T) -> Option<T>
    where
        T: Clone + WasmCompatSend + WasmCompatSync + 'static,
    {
        self.lock().insert(value)
    }

    /// Get a cloned value.
    pub fn get<T>(&self) -> Option<T>
    where
        T: Clone + WasmCompatSend + WasmCompatSync + 'static,
    {
        self.lock().get::<T>().cloned()
    }

    /// Whether a type is present.
    pub fn contains<T>(&self) -> bool
    where
        T: WasmCompatSend + WasmCompatSync + 'static,
    {
        self.lock().contains::<T>()
    }

    /// Remove a value.
    pub fn remove<T>(&self) -> Option<T>
    where
        T: Clone + WasmCompatSend + WasmCompatSync + 'static,
    {
        self.lock().remove::<T>()
    }

    /// Atomically update a value, starting at `Default`.
    pub fn update<T, R>(&self, update: impl FnOnce(&mut T) -> R) -> R
    where
        T: Clone + Default + WasmCompatSend + WasmCompatSync + 'static,
    {
        let mut guard = self.lock();
        let mut value = guard.remove::<T>().unwrap_or_default();
        let result = update(&mut value);
        guard.insert(value);
        result
    }
}

impl std::fmt::Debug for Scratchpad {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scratchpad")
            .field("entries", &self.lock().len())
            .finish()
    }
}

type ToolCallRewriteFrameMap = HashMap<String, Vec<Option<serde_json::Value>>>;

// A nested `HookStack` can terminate after rewriting arguments, but the public
// action only carries the terminal reason. Resolution frames transfer that
// rewrite across the private erased-hook boundary. Call IDs keep concurrently
// executing tool chains isolated, and the frame stack supports arbitrary nesting.
#[derive(Default)]
struct ToolCallRewriteFrames {
    inner: std::sync::Mutex<ToolCallRewriteFrameMap>,
}

impl ToolCallRewriteFrames {
    fn lock(&self) -> std::sync::MutexGuard<'_, ToolCallRewriteFrameMap> {
        self.inner.lock().unwrap_or_else(|error| error.into_inner())
    }

    fn begin(&self, internal_call_id: &str) -> ToolCallResolutionFrame<'_> {
        self.lock()
            .entry(internal_call_id.to_owned())
            .or_default()
            .push(None);
        ToolCallResolutionFrame {
            frames: self,
            internal_call_id: internal_call_id.to_owned(),
            active: true,
        }
    }

    fn record(&self, internal_call_id: &str, rewrite: serde_json::Value) {
        if let Some(frame) = self
            .lock()
            .get_mut(internal_call_id)
            .and_then(|frames| frames.last_mut())
        {
            *frame = Some(rewrite);
        }
    }

    fn finish(&self, internal_call_id: &str) -> Option<serde_json::Value> {
        let mut frames = self.lock();
        let (rewrite, remove_entry) = frames
            .get_mut(internal_call_id)
            .map(|frames| {
                let rewrite = frames.pop().flatten();
                (rewrite, frames.is_empty())
            })
            .unwrap_or((None, false));
        if remove_entry {
            frames.remove(internal_call_id);
        }
        rewrite
    }
}

impl std::fmt::Debug for ToolCallRewriteFrames {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolCallRewriteFrames")
            .finish_non_exhaustive()
    }
}

struct ToolCallResolutionFrame<'a> {
    frames: &'a ToolCallRewriteFrames,
    internal_call_id: String,
    active: bool,
}

impl ToolCallResolutionFrame<'_> {
    fn finish(mut self) -> Option<serde_json::Value> {
        self.active = false;
        self.frames.finish(&self.internal_call_id)
    }
}

impl Drop for ToolCallResolutionFrame<'_> {
    fn drop(&mut self) {
        if self.active {
            self.frames.finish(&self.internal_call_id);
        }
    }
}

/// Run-scoped context supplied to hooks.
#[derive(Debug)]
pub struct HookContext {
    run_id: RunId,
    turn: AtomicUsize,
    is_streaming: bool,
    agent_name: Option<String>,
    scratchpad: Scratchpad,
    tool_call_rewrite_frames: ToolCallRewriteFrames,
}

impl HookContext {
    pub(crate) fn new(is_streaming: bool, agent_name: Option<String>) -> Self {
        Self {
            run_id: RunId::generate(),
            turn: AtomicUsize::new(0),
            is_streaming,
            agent_name,
            scratchpad: Scratchpad::default(),
            tool_call_rewrite_frames: ToolCallRewriteFrames::default(),
        }
    }

    pub(crate) fn resume(
        run_id: String,
        turn: usize,
        is_streaming: bool,
        agent_name: Option<String>,
        scratchpad: Scratchpad,
    ) -> Self {
        Self {
            run_id: RunId::from_stored(run_id),
            turn: AtomicUsize::new(turn),
            is_streaming,
            agent_name,
            scratchpad,
            tool_call_rewrite_frames: ToolCallRewriteFrames::default(),
        }
    }

    pub(crate) fn set_turn(&self, turn: usize) {
        self.turn.store(turn, Ordering::Relaxed);
    }

    /// Stable run identifier.
    pub fn run_id(&self) -> &RunId {
        &self.run_id
    }

    /// Current one-based model-call index.
    pub fn turn(&self) -> usize {
        self.turn.load(Ordering::Relaxed)
    }

    /// Whether the streaming surface is driving this run.
    pub fn is_streaming(&self) -> bool {
        self.is_streaming
    }

    /// Configured agent name.
    pub fn agent_name(&self) -> Option<&str> {
        self.agent_name.as_deref()
    }

    /// Shared run scratchpad.
    pub fn scratchpad(&self) -> &Scratchpad {
        &self.scratchpad
    }

    fn begin_tool_call_resolution(&self, internal_call_id: &str) -> ToolCallResolutionFrame<'_> {
        self.tool_call_rewrite_frames.begin(internal_call_id)
    }

    fn record_tool_call_rewrite(&self, internal_call_id: &str, rewrite: serde_json::Value) {
        self.tool_call_rewrite_frames
            .record(internal_call_id, rewrite);
    }
}

/// Diagnostics for an invalid model-emitted tool call.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct InvalidToolCallContext {
    /// Name emitted by the model.
    pub tool_name: String,
    /// Provider tool-call id, when present.
    pub tool_call_id: Option<String>,
    /// Rig correlation id, when present.
    pub internal_call_id: Option<String>,
    /// Emitted JSON arguments, when present.
    pub args: Option<String>,
    /// Executable tools advertised for the turn.
    pub available_tools: Vec<String>,
    /// Tools permitted by the active tool choice.
    pub allowed_tools: Vec<String>,
    /// Active tool choice.
    pub tool_choice: Option<ToolChoice>,
    /// Diagnostic history including the rejected output.
    pub chat_history: Vec<Message>,
    /// Whether the call came from the streaming path.
    pub is_streaming: bool,
}

/// Completion-call event.
#[derive(Clone, Copy)]
pub struct CompletionCall<'a> {
    /// Prompt for this turn.
    pub prompt: &'a Message,
    /// History preceding the prompt.
    pub history: &'a [Message],
    /// One-based model-call index.
    pub turn: usize,
}

/// Non-streaming completion response event.
pub struct CompletionResponse<'a, M: CompletionModel> {
    /// Prompt sent for this turn.
    pub prompt: &'a Message,
    /// Normalized response and raw provider payload.
    pub response: &'a crate::completion::CompletionResponse<M::Response>,
}

impl<M: CompletionModel> Copy for CompletionResponse<'_, M> {}
impl<M: CompletionModel> Clone for CompletionResponse<'_, M> {
    fn clone(&self) -> Self {
        *self
    }
}

/// Medium-neutral accepted model-turn event.
#[derive(Clone, Copy)]
pub struct ModelTurnFinished<'a> {
    /// One-based model-call index.
    pub turn: usize,
    /// Canonical committed assistant content.
    pub content: &'a OneOrMany<AssistantContent>,
    /// Usage reported for the turn.
    pub usage: Usage,
}

/// Pre-execution tool event.
#[derive(Clone, Copy)]
pub struct ToolCall<'a> {
    /// Tool name.
    pub tool_name: &'a str,
    /// Provider tool-call id.
    pub tool_call_id: Option<&'a str>,
    /// Rig correlation id.
    pub internal_call_id: &'a str,
    /// Effective JSON arguments, including earlier rewrites.
    pub args: &'a str,
}

/// Post-execution tool event.
///
/// `presentation` contains the running presentation rewrite. `raw_result` and
/// `tool_context` always contain the original execution data.
#[derive(Clone, Copy)]
pub struct ToolResultEvent<'a> {
    /// Tool name.
    pub tool_name: &'a str,
    /// Provider tool-call id.
    pub tool_call_id: Option<&'a str>,
    /// Rig correlation id.
    pub internal_call_id: &'a str,
    /// Effective arguments used for execution.
    pub args: &'a str,
    /// Current model-visible presentation, including earlier rewrites.
    pub presentation: &'a ToolOutput,
    /// Immutable raw execution result.
    pub raw_result: &'a ToolResult,
    /// Per-dispatch context containing inbound data and result metadata.
    pub tool_context: &'a ToolContext,
}

/// Streaming text delta.
#[derive(Clone, Copy)]
pub struct TextDelta<'a> {
    /// Newly received text.
    pub delta: &'a str,
    /// Text accumulated for the turn.
    pub aggregated: &'a str,
}

/// Streaming tool-call delta.
#[derive(Clone, Copy)]
pub struct ToolCallDelta<'a> {
    /// Provider tool-call id.
    pub tool_call_id: &'a str,
    /// Rig correlation id.
    pub internal_call_id: &'a str,
    /// Tool name on the first delta.
    pub tool_name: Option<&'a str>,
    /// Newly received argument fragment.
    pub delta: &'a str,
}

/// Streaming response-finish event.
pub struct StreamResponseFinish<'a, M: CompletionModel> {
    /// Prompt sent for this turn.
    pub prompt: &'a Message,
    /// Provider's final streaming response.
    pub response: &'a M::StreamingResponse,
}

impl<M: CompletionModel> Copy for StreamResponseFinish<'_, M> {}
impl<M: CompletionModel> Clone for StreamResponseFinish<'_, M> {
    fn clone(&self) -> Self {
        *self
    }
}

/// Hook event kind used only as an observation performance hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum StepEventKind {
    CompletionCall,
    CompletionResponse,
    ModelTurnFinished,
    InvalidToolCall,
    ToolCall,
    ToolResult,
    TextDelta,
    ToolCallDelta,
    StreamResponseFinish,
}

/// A non-sticky patch applied only to the current turn's completion request.
///
/// A [`HookStack`] merges patches in hook registration order according to these
/// rules:
///
/// - `extra_context` documents are appended in order.
/// - JSON-object `additional_params` values are shallow-merged, with later
///   top-level keys winning; a later non-object value replaces an earlier value.
/// - `active_tools` allow-lists are intersected.
/// - Scalar fields and `history` use last-writer-wins semantics, with a warning
///   when multiple hooks set the same field.
///
/// The merged patch does not mutate the agent's configured baseline and is not
/// carried into subsequent turns.
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct RequestPatch {
    /// Preamble to use instead of the agent's configured preamble for this turn.
    pub preamble: Option<String>,
    /// Sampling temperature to use for this turn.
    pub temperature: Option<f64>,
    /// Maximum output-token count to use for this turn.
    pub max_tokens: Option<u64>,
    /// Tool-choice policy to use for this turn.
    pub tool_choice: Option<ToolChoice>,
    /// Allow-list used to narrow the tools advertised for this turn.
    pub active_tools: Option<Vec<String>>,
    /// Provider-specific request parameters to apply for this turn.
    pub additional_params: Option<serde_json::Value>,
    /// Context documents appended to the request for this turn.
    pub extra_context: Vec<Document>,
    /// Conversation history to use instead of the current history for this turn.
    pub history: Option<Vec<Message>>,
}

fn merge_last_wins<T>(earlier: Option<T>, later: Option<T>, field: &str) -> Option<T> {
    match (earlier, later) {
        (Some(_), Some(later)) => {
            tracing::warn!(
                patch_field = field,
                "two hooks set the same request field; later wins"
            );
            Some(later)
        }
        (earlier, later) => later.or(earlier),
    }
}

impl RequestPatch {
    /// Creates an empty request patch.
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces the agent's configured preamble for this turn.
    pub fn preamble(mut self, value: impl Into<String>) -> Self {
        self.preamble = Some(value.into());
        self
    }

    /// Sets the sampling temperature for this turn.
    pub fn temperature(mut self, value: f64) -> Self {
        self.temperature = Some(value);
        self
    }

    /// Sets the maximum output-token count for this turn.
    pub fn max_tokens(mut self, value: u64) -> Self {
        self.max_tokens = Some(value);
        self
    }

    /// Sets the tool-choice policy for this turn.
    pub fn tool_choice(mut self, value: ToolChoice) -> Self {
        self.tool_choice = Some(value);
        self
    }

    /// Sets the allow-list used to narrow the tools advertised for this turn.
    pub fn active_tools<I, S>(mut self, values: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.active_tools = Some(values.into_iter().map(Into::into).collect());
        self
    }

    /// Sets provider-specific request parameters for this turn.
    ///
    /// When multiple patches provide JSON objects, their top-level keys are
    /// shallow-merged and values from later hooks win.
    pub fn additional_params(mut self, value: serde_json::Value) -> Self {
        self.additional_params = Some(value);
        self
    }

    /// Appends context documents to the request for this turn.
    pub fn extra_context<I>(mut self, values: I) -> Self
    where
        I: IntoIterator<Item = Document>,
    {
        self.extra_context.extend(values);
        self
    }

    /// Appends one context document to the request for this turn.
    pub fn context(mut self, value: Document) -> Self {
        self.extra_context.push(value);
        self
    }

    /// Replaces the conversation history for this turn.
    pub fn history<I>(mut self, values: I) -> Self
    where
        I: IntoIterator<Item = Message>,
    {
        self.history = Some(values.into_iter().collect());
        self
    }

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

    pub(crate) fn merge(mut self, later: Self) -> Self {
        self.extra_context.extend(later.extra_context);
        self.additional_params = match (self.additional_params.take(), later.additional_params) {
            (Some(base), Some(patch)) if base.is_object() && patch.is_object() => {
                Some(json_utils::merge(base, patch))
            }
            (base, patch) => patch.or(base),
        };
        self.preamble = merge_last_wins(self.preamble, later.preamble, "preamble");
        self.temperature = merge_last_wins(self.temperature, later.temperature, "temperature");
        self.max_tokens = merge_last_wins(self.max_tokens, later.max_tokens, "max_tokens");
        self.tool_choice = merge_last_wins(self.tool_choice, later.tool_choice, "tool_choice");
        self.history = merge_last_wins(self.history, later.history, "history");
        self.active_tools = match (self.active_tools.take(), later.active_tools) {
            (Some(earlier), Some(later)) => {
                let later: std::collections::BTreeSet<_> = later.iter().collect();
                Some(
                    earlier
                        .into_iter()
                        .filter(|name| later.contains(name))
                        .collect(),
                )
            }
            (earlier, later) => earlier.or(later),
        };
        self
    }
}

/// Action for completion-call hooks.
#[derive(Debug, Clone, PartialEq)]
pub enum CompletionCallAction {
    /// Send the baseline request.
    Continue,
    /// Merge this per-turn patch into the request.
    Patch(RequestPatch),
    /// Stop the run with a reason.
    Stop(String),
}

impl CompletionCallAction {
    /// Creates an action that sends the request without adding a patch.
    pub fn continue_run() -> Self {
        Self::Continue
    }

    /// Creates an action that applies a per-turn request patch.
    pub fn patch(patch: RequestPatch) -> Self {
        Self::Patch(patch)
    }

    /// Creates an action that stops the run with the supplied reason.
    pub fn stop(reason: impl Into<String>) -> Self {
        Self::Stop(reason.into())
    }
}

/// Action for pre-tool hooks.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolCallAction {
    /// Execute with the current arguments.
    Run,
    /// Execute with replacement arguments.
    Rewrite(serde_json::Value),
    /// Do not execute; return this feedback to the model.
    Skip(String),
    /// Stop the run.
    Stop(String),
}

impl ToolCallAction {
    /// Creates an action that executes the tool with the current arguments.
    pub fn run() -> Self {
        Self::Run
    }

    /// Creates an action that replaces the arguments passed to the tool.
    pub fn rewrite(args: impl Into<serde_json::Value>) -> Self {
        Self::Rewrite(args.into())
    }

    /// Serializes replacement arguments and creates a rewrite action.
    ///
    /// Returns an error when `args` cannot be represented as JSON.
    pub fn try_rewrite<T: serde::Serialize>(args: &T) -> Result<Self, serde_json::Error> {
        Ok(Self::Rewrite(serde_json::to_value(args)?))
    }

    /// Creates an action that skips execution and returns feedback to the model.
    pub fn skip(reason: impl Into<String>) -> Self {
        Self::Skip(reason.into())
    }

    /// Creates an action that stops the run before executing the tool.
    pub fn stop(reason: impl Into<String>) -> Self {
        Self::Stop(reason.into())
    }
}

/// Action for post-tool hooks.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolResultAction {
    /// Keep the current presentation.
    Keep,
    /// Replace the effective presentation sent to the model and result-content
    /// telemetry.
    Rewrite(ToolOutput),
    /// Stop the run.
    Stop(String),
}

impl ToolResultAction {
    /// Creates an action that preserves the current model-visible presentation.
    pub fn keep() -> Self {
        Self::Keep
    }

    /// Creates an action that replaces the effective presentation sent to the
    /// model and result-content telemetry.
    ///
    /// The tool's raw structured result remains unchanged.
    pub fn rewrite(result: impl Into<String>) -> Self {
        Self::Rewrite(ToolOutput::text(result))
    }

    /// Creates an action that replaces the effective model and telemetry
    /// presentation with explicit structured or multimodal output.
    pub fn rewrite_output(output: ToolOutput) -> Self {
        Self::Rewrite(output)
    }

    /// Creates an action that stops the run after result handling.
    pub fn stop(reason: impl Into<String>) -> Self {
        Self::Stop(reason.into())
    }
}

/// Action for invalid-tool-call hooks and manual invalid-call resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvalidToolCallAction {
    /// Preserve fail-fast behavior.
    Fail,
    /// Retry the model with corrective feedback.
    Retry {
        /// Feedback appended for the retry.
        feedback: String,
    },
    /// Repair the emitted tool name.
    Repair {
        /// Replacement registered tool name.
        tool_name: String,
    },
    /// Treat the invalid call as skipped.
    Skip {
        /// Synthetic model feedback.
        reason: String,
    },
    /// Stop the run.
    Stop {
        /// Stop reason.
        reason: String,
    },
}

impl InvalidToolCallAction {
    /// Creates an action that preserves fail-fast invalid-call handling.
    pub fn fail() -> Self {
        Self::Fail
    }

    /// Creates an action that retries the model with corrective feedback.
    pub fn retry(feedback: impl Into<String>) -> Self {
        Self::Retry {
            feedback: feedback.into(),
        }
    }

    /// Creates an action that replaces the invalid tool name.
    pub fn repair(tool_name: impl Into<String>) -> Self {
        Self::Repair {
            tool_name: tool_name.into(),
        }
    }

    /// Creates an action that treats the invalid call as skipped.
    pub fn skip(reason: impl Into<String>) -> Self {
        Self::Skip {
            reason: reason.into(),
        }
    }

    /// Creates an action that stops the run with the supplied reason.
    pub fn stop(reason: impl Into<String>) -> Self {
        Self::Stop {
            reason: reason.into(),
        }
    }
}

/// Action for observe-only lifecycle events.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObservationAction {
    /// Continue the run.
    Continue,
    /// Stop the run.
    Stop(String),
}

impl ObservationAction {
    /// Creates an action that continues the run.
    pub fn continue_run() -> Self {
        Self::Continue
    }

    /// Creates an action that stops the run with the supplied reason.
    pub fn stop(reason: impl Into<String>) -> Self {
        Self::Stop(reason.into())
    }
}

/// Per-run lifecycle observer and steerer.
pub trait AgentHook<M>: WasmCompatSend + WasmCompatSync
where
    M: CompletionModel,
{
    /// Runs before a completion request is sent.
    ///
    /// Return a per-turn patch, continue without one, or stop the run. Patches
    /// from a [`HookStack`] are merged in hook registration order.
    fn on_completion_call(
        &self,
        _ctx: &HookContext,
        _event: CompletionCall<'_>,
    ) -> impl Future<Output = CompletionCallAction> + WasmCompatSend {
        async { CompletionCallAction::Continue }
    }

    /// Observes a completed model response.
    ///
    /// The default action continues the run.
    fn on_completion_response(
        &self,
        _ctx: &HookContext,
        _event: CompletionResponse<'_, M>,
    ) -> impl Future<Output = ObservationAction> + WasmCompatSend {
        async { ObservationAction::Continue }
    }

    /// Observes the content and usage produced at the end of a model turn.
    ///
    /// The default action continues the run.
    fn on_model_turn_finished(
        &self,
        _ctx: &HookContext,
        _event: ModelTurnFinished<'_>,
    ) -> impl Future<Output = ObservationAction> + WasmCompatSend {
        async { ObservationAction::Continue }
    }

    /// Resolves a model-emitted tool call that cannot be dispatched as written.
    ///
    /// The call may be failed, retried, repaired, skipped, or used to stop the
    /// run. Return `None` to leave the decision to a later hook. If every hook
    /// in a [`HookStack`] returns `None`, the agent preserves fail-fast
    /// behavior.
    fn on_invalid_tool_call(
        &self,
        _ctx: &HookContext,
        _event: &InvalidToolCallContext,
    ) -> impl Future<Output = Option<InvalidToolCallAction>> + WasmCompatSend {
        async { None }
    }

    /// Runs before a valid tool call is executed.
    ///
    /// The hook may rewrite the current arguments, skip execution, or stop the
    /// run. Rewrites in a [`HookStack`] are passed to subsequent hooks. The
    /// default action executes with the current arguments.
    fn on_tool_call(
        &self,
        _ctx: &HookContext,
        _event: ToolCall<'_>,
    ) -> impl Future<Output = ToolCallAction> + WasmCompatSend {
        async { ToolCallAction::Run }
    }

    /// Runs after a tool call resolves and before its presentation is sent to the model.
    ///
    /// This includes framework-skipped calls whose tool body did not execute.
    /// Rewrites affect the model-visible presentation and result-content
    /// telemetry, but not the raw structured result or execution-outcome
    /// metadata. A stop omits result content from telemetry. The default action
    /// keeps the current presentation.
    fn on_tool_result(
        &self,
        _ctx: &HookContext,
        _event: ToolResultEvent<'_>,
    ) -> impl Future<Output = ToolResultAction> + WasmCompatSend {
        async { ToolResultAction::Keep }
    }

    /// Observes a text delta from a streaming response.
    ///
    /// The default action continues the run.
    fn on_text_delta(
        &self,
        _ctx: &HookContext,
        _event: TextDelta<'_>,
    ) -> impl Future<Output = ObservationAction> + WasmCompatSend {
        async { ObservationAction::Continue }
    }

    /// Observes an argument delta for a streaming tool call.
    ///
    /// The default action continues the run.
    fn on_tool_call_delta(
        &self,
        _ctx: &HookContext,
        _event: ToolCallDelta<'_>,
    ) -> impl Future<Output = ObservationAction> + WasmCompatSend {
        async { ObservationAction::Continue }
    }

    /// Observes the provider's final streaming response.
    ///
    /// The default action continues the run.
    fn on_stream_response_finish(
        &self,
        _ctx: &HookContext,
        _event: StreamResponseFinish<'_, M>,
    ) -> impl Future<Output = ObservationAction> + WasmCompatSend {
        async { ObservationAction::Continue }
    }

    /// Observation interest hint, primarily for high-frequency deltas.
    fn observes(&self, _kind: StepEventKind) -> bool {
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
        ctx: &'a HookContext,
        event: CompletionCall<'a>,
    ) -> WasmBoxedFuture<'a, CompletionCallAction>
    where
        M: 'a;
    fn completion_response<'a>(
        &'a self,
        ctx: &'a HookContext,
        event: CompletionResponse<'a, M>,
    ) -> WasmBoxedFuture<'a, ObservationAction>
    where
        M: 'a;
    fn model_turn_finished<'a>(
        &'a self,
        ctx: &'a HookContext,
        event: ModelTurnFinished<'a>,
    ) -> WasmBoxedFuture<'a, ObservationAction>
    where
        M: 'a;
    fn invalid_tool_call<'a>(
        &'a self,
        ctx: &'a HookContext,
        event: &'a InvalidToolCallContext,
    ) -> WasmBoxedFuture<'a, Option<InvalidToolCallAction>>
    where
        M: 'a;
    fn tool_call<'a>(
        &'a self,
        ctx: &'a HookContext,
        event: ToolCall<'a>,
    ) -> WasmBoxedFuture<'a, (ToolCallAction, Option<serde_json::Value>)>
    where
        M: 'a;
    fn tool_result<'a>(
        &'a self,
        ctx: &'a HookContext,
        event: ToolResultEvent<'a>,
    ) -> WasmBoxedFuture<'a, ToolResultAction>
    where
        M: 'a;
    fn text_delta<'a>(
        &'a self,
        ctx: &'a HookContext,
        event: TextDelta<'a>,
    ) -> WasmBoxedFuture<'a, ObservationAction>
    where
        M: 'a;
    fn tool_call_delta<'a>(
        &'a self,
        ctx: &'a HookContext,
        event: ToolCallDelta<'a>,
    ) -> WasmBoxedFuture<'a, ObservationAction>
    where
        M: 'a;
    fn stream_response_finish<'a>(
        &'a self,
        ctx: &'a HookContext,
        event: StreamResponseFinish<'a, M>,
    ) -> WasmBoxedFuture<'a, ObservationAction>
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
        ctx: &'a HookContext,
        event: CompletionCall<'a>,
    ) -> WasmBoxedFuture<'a, CompletionCallAction>
    where
        M: 'a,
    {
        Box::pin(self.on_completion_call(ctx, event))
    }
    fn completion_response<'a>(
        &'a self,
        ctx: &'a HookContext,
        event: CompletionResponse<'a, M>,
    ) -> WasmBoxedFuture<'a, ObservationAction>
    where
        M: 'a,
    {
        Box::pin(self.on_completion_response(ctx, event))
    }
    fn model_turn_finished<'a>(
        &'a self,
        ctx: &'a HookContext,
        event: ModelTurnFinished<'a>,
    ) -> WasmBoxedFuture<'a, ObservationAction>
    where
        M: 'a,
    {
        Box::pin(self.on_model_turn_finished(ctx, event))
    }
    fn invalid_tool_call<'a>(
        &'a self,
        ctx: &'a HookContext,
        event: &'a InvalidToolCallContext,
    ) -> WasmBoxedFuture<'a, Option<InvalidToolCallAction>>
    where
        M: 'a,
    {
        Box::pin(self.on_invalid_tool_call(ctx, event))
    }
    fn tool_call<'a>(
        &'a self,
        ctx: &'a HookContext,
        event: ToolCall<'a>,
    ) -> WasmBoxedFuture<'a, (ToolCallAction, Option<serde_json::Value>)>
    where
        M: 'a,
    {
        Box::pin(async move {
            // Only `on_tool_call` is public dispatch. A nested `HookStack`
            // records terminal-path rewrite state into this private frame.
            let frame = ctx.begin_tool_call_resolution(event.internal_call_id);
            let action = self.on_tool_call(ctx, event).await;
            (action, frame.finish())
        })
    }
    fn tool_result<'a>(
        &'a self,
        ctx: &'a HookContext,
        event: ToolResultEvent<'a>,
    ) -> WasmBoxedFuture<'a, ToolResultAction>
    where
        M: 'a,
    {
        Box::pin(self.on_tool_result(ctx, event))
    }
    fn text_delta<'a>(
        &'a self,
        ctx: &'a HookContext,
        event: TextDelta<'a>,
    ) -> WasmBoxedFuture<'a, ObservationAction>
    where
        M: 'a,
    {
        Box::pin(self.on_text_delta(ctx, event))
    }
    fn tool_call_delta<'a>(
        &'a self,
        ctx: &'a HookContext,
        event: ToolCallDelta<'a>,
    ) -> WasmBoxedFuture<'a, ObservationAction>
    where
        M: 'a,
    {
        Box::pin(self.on_tool_call_delta(ctx, event))
    }
    fn stream_response_finish<'a>(
        &'a self,
        ctx: &'a HookContext,
        event: StreamResponseFinish<'a, M>,
    ) -> WasmBoxedFuture<'a, ObservationAction>
    where
        M: 'a,
    {
        Box::pin(self.on_stream_response_finish(ctx, event))
    }
    fn observes(&self, kind: StepEventKind) -> bool {
        AgentHook::observes(self, kind)
    }
}

/// Ordered composable hook stack.
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
    /// Creates an empty hook stack.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a hook stack containing `hook`.
    pub fn with<H: AgentHook<M> + 'static>(hook: H) -> Self {
        let mut stack = Self::new();
        stack.push(hook);
        stack
    }

    /// Appends a hook to the end of the stack's registration order.
    pub fn push<H: AgentHook<M> + 'static>(&mut self, hook: H) {
        self.hooks.push(Arc::new(hook));
    }

    /// Returns `true` when the stack contains no hooks.
    pub fn is_empty(&self) -> bool {
        self.hooks.is_empty()
    }

    /// Returns the number of hooks in the stack.
    pub fn len(&self) -> usize {
        self.hooks.len()
    }

    /// Resolve the hook chain while retaining a rewrite accumulated before a
    /// terminal action so the runner can report the effective arguments.
    pub(crate) async fn resolve_tool_call(
        &self,
        ctx: &HookContext,
        event: ToolCall<'_>,
    ) -> (ToolCallAction, Option<serde_json::Value>) {
        let mut effective = None;
        for hook in &self.hooks {
            let rewritten = effective.as_ref().map(json_utils::serialize_json_value);
            let current = ToolCall {
                args: rewritten.as_deref().unwrap_or(event.args),
                ..event
            };
            let (action, salvaged) = hook.tool_call(ctx, current).await;
            if let Some(value) = salvaged {
                effective = Some(value);
            }
            match action {
                ToolCallAction::Run => {}
                ToolCallAction::Rewrite(value) => effective = Some(value),
                other => return (other, effective),
            }
        }
        match effective {
            Some(value) => (ToolCallAction::Rewrite(value), None),
            None => (ToolCallAction::Run, None),
        }
    }
}

async fn first_stop<I>(futures: I) -> ObservationAction
where
    I: IntoIterator<Item = ObservationAction>,
{
    for action in futures {
        if !matches!(action, ObservationAction::Continue) {
            return action;
        }
    }
    ObservationAction::Continue
}

impl<M: CompletionModel> AgentHook<M> for HookStack<M> {
    async fn on_completion_call(
        &self,
        ctx: &HookContext,
        event: CompletionCall<'_>,
    ) -> CompletionCallAction {
        let mut merged: Option<RequestPatch> = None;
        for hook in &self.hooks {
            match hook.completion_call(ctx, event).await {
                CompletionCallAction::Continue => {}
                CompletionCallAction::Patch(patch) => {
                    merged = Some(merged.map_or(patch.clone(), |value| value.merge(patch)))
                }
                stop @ CompletionCallAction::Stop(_) => return stop,
            }
        }
        match merged {
            Some(patch) if !patch.is_empty() => CompletionCallAction::Patch(patch),
            _ => CompletionCallAction::Continue,
        }
    }

    async fn on_completion_response(
        &self,
        ctx: &HookContext,
        event: CompletionResponse<'_, M>,
    ) -> ObservationAction {
        let mut actions = Vec::new();
        for hook in &self.hooks {
            let action = hook.completion_response(ctx, event).await;
            let stop = !matches!(action, ObservationAction::Continue);
            actions.push(action);
            if stop {
                break;
            }
        }
        first_stop(actions).await
    }
    async fn on_model_turn_finished(
        &self,
        ctx: &HookContext,
        event: ModelTurnFinished<'_>,
    ) -> ObservationAction {
        for hook in &self.hooks {
            let action = hook.model_turn_finished(ctx, event).await;
            if !matches!(action, ObservationAction::Continue) {
                return action;
            }
        }
        ObservationAction::Continue
    }
    async fn on_invalid_tool_call(
        &self,
        ctx: &HookContext,
        event: &InvalidToolCallContext,
    ) -> Option<InvalidToolCallAction> {
        for hook in &self.hooks {
            if let Some(action) = hook.invalid_tool_call(ctx, event).await {
                return Some(action);
            }
        }
        None
    }
    async fn on_tool_call(&self, ctx: &HookContext, event: ToolCall<'_>) -> ToolCallAction {
        let internal_call_id = event.internal_call_id;
        let (action, salvaged) = self.resolve_tool_call(ctx, event).await;
        // This is a no-op for direct calls. Under private erased dispatch it
        // returns a nested stack's terminal-path rewrite to its parent stack.
        if let Some(rewrite) = salvaged {
            ctx.record_tool_call_rewrite(internal_call_id, rewrite);
        }
        action
    }
    async fn on_tool_result(
        &self,
        ctx: &HookContext,
        event: ToolResultEvent<'_>,
    ) -> ToolResultAction {
        let mut effective: Option<ToolOutput> = None;
        for hook in &self.hooks {
            let current = ToolResultEvent {
                presentation: effective.as_ref().unwrap_or(event.presentation),
                ..event
            };
            match hook.tool_result(ctx, current).await {
                ToolResultAction::Keep => {}
                ToolResultAction::Rewrite(value) => effective = Some(value),
                stop @ ToolResultAction::Stop(_) => return stop,
            }
        }
        effective.map_or(ToolResultAction::Keep, ToolResultAction::Rewrite)
    }
    async fn on_text_delta(&self, ctx: &HookContext, event: TextDelta<'_>) -> ObservationAction {
        for hook in &self.hooks {
            let action = hook.text_delta(ctx, event).await;
            if !matches!(action, ObservationAction::Continue) {
                return action;
            }
        }
        ObservationAction::Continue
    }
    async fn on_tool_call_delta(
        &self,
        ctx: &HookContext,
        event: ToolCallDelta<'_>,
    ) -> ObservationAction {
        for hook in &self.hooks {
            let action = hook.tool_call_delta(ctx, event).await;
            if !matches!(action, ObservationAction::Continue) {
                return action;
            }
        }
        ObservationAction::Continue
    }
    async fn on_stream_response_finish(
        &self,
        ctx: &HookContext,
        event: StreamResponseFinish<'_, M>,
    ) -> ObservationAction {
        for hook in &self.hooks {
            let action = hook.stream_response_finish(ctx, event).await;
            if !matches!(action, ObservationAction::Continue) {
                return action;
            }
        }
        ObservationAction::Continue
    }
    fn observes(&self, kind: StepEventKind) -> bool {
        self.hooks.iter().any(|hook| hook.observes(kind))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        test_utils::MockCompletionModel,
        tool::{ToolErrorKind, ToolExecutionError},
    };

    struct Patcher(f64);
    impl AgentHook<MockCompletionModel> for Patcher {
        async fn on_completion_call(
            &self,
            _ctx: &HookContext,
            _event: CompletionCall<'_>,
        ) -> CompletionCallAction {
            CompletionCallAction::patch(RequestPatch::new().temperature(self.0))
        }
    }

    #[tokio::test]
    async fn nested_completion_patches_compose() {
        let inner = HookStack::with(Patcher(0.1));
        let mut outer = HookStack::with(inner);
        outer.push(Patcher(0.2));
        let prompt = Message::user("hi");
        let action = outer
            .on_completion_call(
                &HookContext::new(false, None),
                CompletionCall {
                    prompt: &prompt,
                    history: &[],
                    turn: 1,
                },
            )
            .await;
        assert!(matches!(
            action,
            CompletionCallAction::Patch(RequestPatch {
                temperature: Some(0.2),
                ..
            })
        ));
    }

    #[derive(Clone)]
    struct CallRewriter {
        seen: Arc<std::sync::Mutex<Vec<String>>>,
        replacement: serde_json::Value,
    }

    impl AgentHook<MockCompletionModel> for CallRewriter {
        async fn on_tool_call(&self, _ctx: &HookContext, event: ToolCall<'_>) -> ToolCallAction {
            self.seen.lock().unwrap().push(event.args.to_string());
            ToolCallAction::rewrite(self.replacement.clone())
        }
    }

    #[tokio::test]
    async fn tool_call_rewrites_chain_in_registration_order() {
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut stack = HookStack::with(CallRewriter {
            seen: seen.clone(),
            replacement: serde_json::json!({"step": 1}),
        });
        stack.push(CallRewriter {
            seen: seen.clone(),
            replacement: serde_json::json!({"step": 2}),
        });

        let action = stack
            .on_tool_call(
                &HookContext::new(false, None),
                ToolCall {
                    tool_name: "tool",
                    tool_call_id: Some("provider-id"),
                    internal_call_id: "internal-id",
                    args: r#"{"step":0}"#,
                },
            )
            .await;

        assert_eq!(
            *seen.lock().unwrap(),
            vec![r#"{"step":0}"#.to_string(), r#"{"step":1}"#.to_string()]
        );
        assert_eq!(
            action,
            ToolCallAction::rewrite(serde_json::json!({"step": 2}))
        );
    }

    #[derive(Clone)]
    struct ResultRewriter {
        seen: Arc<std::sync::Mutex<Vec<(String, ToolErrorKind, String)>>>,
        replacement: String,
    }

    impl AgentHook<MockCompletionModel> for ResultRewriter {
        async fn on_tool_result(
            &self,
            _ctx: &HookContext,
            event: ToolResultEvent<'_>,
        ) -> ToolResultAction {
            self.seen.lock().unwrap().push((
                event.presentation.render(),
                event.raw_result.error().unwrap().kind(),
                event.tool_context.result::<String>().unwrap().clone(),
            ));
            ToolResultAction::rewrite(self.replacement.clone())
        }
    }

    #[tokio::test]
    async fn result_rewrites_chain_without_mutating_raw_result_or_context() {
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut stack = HookStack::with(ResultRewriter {
            seen: seen.clone(),
            replacement: "redacted".into(),
        });
        stack.push(ResultRewriter {
            seen: seen.clone(),
            replacement: "truncated".into(),
        });
        let raw = ToolResult::failed(ToolExecutionError::timeout("raw failure"));
        let mut context = ToolContext::new();
        context.insert_result("request-metadata".to_string());

        let action = stack
            .on_tool_result(
                &HookContext::new(false, None),
                ToolResultEvent {
                    tool_name: "tool",
                    tool_call_id: None,
                    internal_call_id: "internal-id",
                    args: "{}",
                    presentation: raw.output(),
                    raw_result: &raw,
                    tool_context: &context,
                },
            )
            .await;

        assert_eq!(action, ToolResultAction::rewrite("truncated"));
        assert_eq!(
            *seen.lock().unwrap(),
            vec![
                (
                    "raw failure".into(),
                    ToolErrorKind::Timeout,
                    "request-metadata".into()
                ),
                (
                    "redacted".into(),
                    ToolErrorKind::Timeout,
                    "request-metadata".into()
                ),
            ]
        );
        assert_eq!(raw.output().as_text(), Some("raw failure"));
        assert_eq!(
            context.result::<String>().map(String::as_str),
            Some("request-metadata")
        );
    }

    struct StopThenCount {
        stop: bool,
        calls: Arc<AtomicUsize>,
    }

    impl AgentHook<MockCompletionModel> for StopThenCount {
        async fn on_tool_result(
            &self,
            _ctx: &HookContext,
            _event: ToolResultEvent<'_>,
        ) -> ToolResultAction {
            self.calls.fetch_add(1, Ordering::Relaxed);
            if self.stop {
                ToolResultAction::stop("terminal")
            } else {
                ToolResultAction::keep()
            }
        }
    }

    #[tokio::test]
    async fn terminal_result_action_short_circuits_later_hooks() {
        let calls = Arc::new(AtomicUsize::new(0));
        let mut stack = HookStack::with(StopThenCount {
            stop: true,
            calls: calls.clone(),
        });
        stack.push(StopThenCount {
            stop: false,
            calls: calls.clone(),
        });
        let raw = ToolResult::success(ToolOutput::text("ok"));
        let context = ToolContext::new();
        let action = stack
            .on_tool_result(
                &HookContext::new(false, None),
                ToolResultEvent {
                    tool_name: "tool",
                    tool_call_id: None,
                    internal_call_id: "internal-id",
                    args: "{}",
                    presentation: raw.output(),
                    raw_result: &raw,
                    tool_context: &context,
                },
            )
            .await;

        assert_eq!(action, ToolResultAction::stop("terminal"));
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }
}

#[cfg(test)]
mod migrated_tests {
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    };

    use super::*;
    use crate::test_utils::MockCompletionModel;
    use serde_json::{Value, json};

    type M = MockCompletionModel;

    fn ctx() -> HookContext {
        HookContext::new(false, Some("test-agent".to_string()))
    }

    struct ToolRecorder {
        label: u32,
        log: Arc<Mutex<Vec<u32>>>,
        stop: bool,
    }
    impl AgentHook<M> for ToolRecorder {
        async fn on_tool_call(&self, _ctx: &HookContext, _event: ToolCall<'_>) -> ToolCallAction {
            self.log.lock().expect("log").push(self.label);
            if self.stop {
                ToolCallAction::stop("stop")
            } else {
                ToolCallAction::run()
            }
        }
    }

    struct ObservationRecorder {
        label: u32,
        log: Arc<Mutex<Vec<u32>>>,
        stop: bool,
    }
    impl AgentHook<M> for ObservationRecorder {
        async fn on_text_delta(
            &self,
            _ctx: &HookContext,
            _event: TextDelta<'_>,
        ) -> ObservationAction {
            self.log.lock().expect("log").push(self.label);
            if self.stop {
                ObservationAction::stop("stop")
            } else {
                ObservationAction::continue_run()
            }
        }
    }

    struct ObservesOnly(StepEventKind);
    impl AgentHook<M> for ObservesOnly {
        fn observes(&self, kind: StepEventKind) -> bool {
            kind == self.0
        }
    }

    struct InvalidResponder {
        action: InvalidToolCallAction,
        calls: Arc<AtomicUsize>,
    }
    impl AgentHook<M> for InvalidResponder {
        async fn on_invalid_tool_call(
            &self,
            _ctx: &HookContext,
            _event: &InvalidToolCallContext,
        ) -> Option<InvalidToolCallAction> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Some(self.action.clone())
        }
    }

    struct Patcher {
        label: u32,
        log: Arc<Mutex<Vec<u32>>>,
        patch: RequestPatch,
        stop: bool,
    }
    impl AgentHook<M> for Patcher {
        async fn on_completion_call(
            &self,
            _ctx: &HookContext,
            _event: CompletionCall<'_>,
        ) -> CompletionCallAction {
            self.log.lock().expect("log").push(self.label);
            if self.stop {
                CompletionCallAction::stop("stop")
            } else {
                CompletionCallAction::patch(self.patch.clone())
            }
        }
    }

    fn tool_call_event() -> ToolCall<'static> {
        ToolCall {
            tool_name: "add",
            tool_call_id: Some("tc1"),
            internal_call_id: "ic1",
            args: "{}",
        }
    }
    fn completion_call_event() -> CompletionCall<'static> {
        static PROMPT: std::sync::OnceLock<crate::message::Message> = std::sync::OnceLock::new();
        CompletionCall {
            prompt: PROMPT.get_or_init(|| crate::message::Message::user("hi")),
            history: &[],
            turn: 1,
        }
    }

    fn invalid_tool_call_context() -> InvalidToolCallContext {
        InvalidToolCallContext {
            tool_name: "unknown".into(),
            tool_call_id: Some("tc1".into()),
            internal_call_id: Some("ic1".into()),
            args: Some("{}".into()),
            available_tools: vec!["add".into()],
            allowed_tools: vec!["add".into()],
            tool_choice: None,
            chat_history: vec![],
            is_streaming: false,
        }
    }

    #[tokio::test]
    async fn runs_hooks_in_registration_order_and_consults_all_on_continue() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut stack = HookStack::<M>::with(ToolRecorder {
            label: 1,
            log: log.clone(),
            stop: false,
        });
        stack.push(ToolRecorder {
            label: 2,
            log: log.clone(),
            stop: false,
        });
        assert_eq!(
            stack.on_tool_call(&ctx(), tool_call_event()).await,
            ToolCallAction::run()
        );
        assert_eq!(*log.lock().unwrap(), vec![1, 2]);
    }

    #[tokio::test]
    async fn first_stop_short_circuits_on_chained_tool_call() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut stack = HookStack::<M>::with(ToolRecorder {
            label: 1,
            log: log.clone(),
            stop: true,
        });
        stack.push(ToolRecorder {
            label: 2,
            log: log.clone(),
            stop: false,
        });
        assert!(matches!(
            stack.on_tool_call(&ctx(), tool_call_event()).await,
            ToolCallAction::Stop(_)
        ));
        assert_eq!(*log.lock().unwrap(), vec![1]);
    }

    #[tokio::test]
    async fn first_stop_short_circuits_observation() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut stack = HookStack::<M>::with(ObservationRecorder {
            label: 1,
            log: log.clone(),
            stop: true,
        });
        stack.push(ObservationRecorder {
            label: 2,
            log: log.clone(),
            stop: false,
        });
        assert!(matches!(
            stack
                .on_text_delta(
                    &ctx(),
                    TextDelta {
                        delta: "hi",
                        aggregated: "hi"
                    }
                )
                .await,
            ObservationAction::Stop(_)
        ));
        assert_eq!(*log.lock().unwrap(), vec![1]);
    }

    #[tokio::test]
    async fn explicit_fail_short_circuits_later_invalid_tool_hooks() {
        let fail_calls = Arc::new(AtomicUsize::new(0));
        let retry_calls = Arc::new(AtomicUsize::new(0));
        let mut stack = HookStack::<M>::with(InvalidResponder {
            action: InvalidToolCallAction::fail(),
            calls: fail_calls.clone(),
        });
        stack.push(InvalidResponder {
            action: InvalidToolCallAction::retry("try another tool"),
            calls: retry_calls.clone(),
        });

        let action = stack
            .on_invalid_tool_call(&ctx(), &invalid_tool_call_context())
            .await;

        assert_eq!(action, Some(InvalidToolCallAction::fail()));
        assert_eq!(fail_calls.load(Ordering::Relaxed), 1);
        assert_eq!(retry_calls.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn no_invalid_tool_decision_defers_to_later_hooks() {
        let retry_calls = Arc::new(AtomicUsize::new(0));
        let mut stack = HookStack::<M>::with(());
        stack.push(InvalidResponder {
            action: InvalidToolCallAction::retry("try another tool"),
            calls: retry_calls.clone(),
        });

        let action = stack
            .on_invalid_tool_call(&ctx(), &invalid_tool_call_context())
            .await;

        assert_eq!(
            action,
            Some(InvalidToolCallAction::retry("try another tool"))
        );
        assert_eq!(retry_calls.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn completion_patches_accumulate_and_stop_discards_prior_patch() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut stack = HookStack::<M>::with(Patcher {
            label: 1,
            log: log.clone(),
            patch: RequestPatch::new().temperature(0.1),
            stop: false,
        });
        stack.push(Patcher {
            label: 2,
            log: log.clone(),
            patch: RequestPatch::new().max_tokens(256),
            stop: false,
        });
        match stack
            .on_completion_call(&ctx(), completion_call_event())
            .await
        {
            CompletionCallAction::Patch(p) => {
                assert_eq!(p.temperature, Some(0.1));
                assert_eq!(p.max_tokens, Some(256));
            }
            other => panic!("expected patch, got {other:?}"),
        }
        assert_eq!(*log.lock().unwrap(), vec![1, 2]);
        let mut stopped = HookStack::<M>::with(Patcher {
            label: 3,
            log: log.clone(),
            patch: RequestPatch::new(),
            stop: true,
        });
        stopped.push(Patcher {
            label: 4,
            log: log.clone(),
            patch: RequestPatch::new(),
            stop: false,
        });
        assert!(matches!(
            stopped
                .on_completion_call(&ctx(), completion_call_event())
                .await,
            CompletionCallAction::Stop(_)
        ));
        assert_eq!(*log.lock().unwrap(), vec![1, 2, 3]);
    }

    #[tokio::test]
    async fn nested_stack_composes_patches() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut inner = HookStack::<M>::with(Patcher {
            label: 1,
            log: log.clone(),
            patch: RequestPatch::new().temperature(0.2),
            stop: false,
        });
        inner.push(Patcher {
            label: 2,
            log: log.clone(),
            patch: RequestPatch::new().max_tokens(128),
            stop: false,
        });
        let mut outer = HookStack::<M>::with(inner);
        outer.push(Patcher {
            label: 3,
            log: log.clone(),
            patch: RequestPatch::new().preamble("outer"),
            stop: false,
        });
        match outer
            .on_completion_call(&ctx(), completion_call_event())
            .await
        {
            CompletionCallAction::Patch(p) => {
                assert_eq!(p.temperature, Some(0.2));
                assert_eq!(p.max_tokens, Some(128));
                assert_eq!(p.preamble.as_deref(), Some("outer"));
            }
            other => panic!("expected patch, got {other:?}"),
        }
        assert_eq!(*log.lock().unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn stack_observes_is_the_or_of_members() {
        let mut stack = HookStack::<M>::with(ObservesOnly(StepEventKind::ToolCall));
        stack.push(ObservesOnly(StepEventKind::ToolResult));
        assert!(<HookStack<M> as AgentHook<M>>::observes(
            &stack,
            StepEventKind::ToolCall
        ));
        assert!(<HookStack<M> as AgentHook<M>>::observes(
            &stack,
            StepEventKind::ToolResult
        ));
        assert!(!<HookStack<M> as AgentHook<M>>::observes(
            &stack,
            StepEventKind::TextDelta
        ));
    }

    #[test]
    fn empty_stack_observes_nothing() {
        let empty = HookStack::<M>::new();
        assert!(empty.is_empty());
        assert!(!<HookStack<M> as AgentHook<M>>::observes(
            &empty,
            StepEventKind::ToolCall
        ));
    }

    #[test]
    fn unit_hook_observes_no_event_kind() {
        for kind in [
            StepEventKind::CompletionCall,
            StepEventKind::CompletionResponse,
            StepEventKind::ModelTurnFinished,
            StepEventKind::InvalidToolCall,
            StepEventKind::ToolCall,
            StepEventKind::ToolResult,
            StepEventKind::TextDelta,
            StepEventKind::ToolCallDelta,
            StepEventKind::StreamResponseFinish,
        ] {
            assert!(!<() as AgentHook<M>>::observes(&(), kind));
        }
    }

    fn doc(id: &str) -> crate::completion::Document {
        crate::completion::Document {
            id: id.into(),
            text: String::new(),
            additional_props: Default::default(),
        }
    }

    #[test]
    fn merge_appends_extra_context_in_order() {
        let merged = RequestPatch::new()
            .context(doc("a"))
            .merge(RequestPatch::new().context(doc("b")));
        assert_eq!(
            merged
                .extra_context
                .iter()
                .map(|d| d.id.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "b"]
        );
    }

    #[test]
    fn merge_shallow_merges_additional_params_later_wins() {
        let merged = RequestPatch::new()
            .additional_params(json!({"x":1,"y":2}))
            .merge(RequestPatch::new().additional_params(json!({"y":3,"z":4})));
        assert_eq!(merged.additional_params, Some(json!({"x":1,"y":3,"z":4})));
    }

    #[test]
    fn merge_scalar_last_writer_wins() {
        assert_eq!(
            RequestPatch::new()
                .temperature(0.1)
                .merge(RequestPatch::new().temperature(0.9))
                .temperature,
            Some(0.9)
        );
    }

    #[test]
    fn merge_active_tools_intersects() {
        let merged = RequestPatch::new()
            .active_tools(["add", "sub"])
            .merge(RequestPatch::new().active_tools(["sub", "mul"]));
        assert_eq!(merged.active_tools, Some(vec!["sub".into()]));
    }

    #[test]
    fn merge_active_tools_empty_intersection_yields_empty() {
        assert_eq!(
            RequestPatch::new()
                .active_tools(["a"])
                .merge(RequestPatch::new().active_tools(["b"]))
                .active_tools,
            Some(vec![])
        );
    }

    #[test]
    fn scratchpad_insert_get_update_remove() {
        #[derive(Clone, Default, Debug, PartialEq)]
        struct Count(u32);
        let pad = Scratchpad::default();
        pad.update(|c: &mut Count| c.0 += 1);
        pad.update(|c: &mut Count| c.0 += 1);
        assert_eq!(pad.get::<Count>(), Some(Count(2)));
        assert_eq!(pad.remove::<Count>(), Some(Count(2)));
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
        let context = HookContext::new(true, Some("agent".into()));
        assert!(context.is_streaming());
        assert_eq!(context.agent_name(), Some("agent"));
        context.set_turn(3);
        assert_eq!(context.turn(), 3);
        assert!(!context.run_id().as_str().is_empty());
    }

    struct RewriteHook(Value);
    impl AgentHook<M> for RewriteHook {
        async fn on_tool_call(&self, _: &HookContext, _: ToolCall<'_>) -> ToolCallAction {
            ToolCallAction::rewrite(self.0.clone())
        }
    }
    struct SkipHook;
    impl AgentHook<M> for SkipHook {
        async fn on_tool_call(&self, _: &HookContext, _: ToolCall<'_>) -> ToolCallAction {
            ToolCallAction::skip("denied")
        }
    }
    struct StopHook;
    impl AgentHook<M> for StopHook {
        async fn on_tool_call(&self, _: &HookContext, _: ToolCall<'_>) -> ToolCallAction {
            ToolCallAction::stop("stop")
        }
    }
    #[derive(Clone, Default)]
    struct ArgsSpy(Arc<Mutex<Vec<String>>>);
    impl AgentHook<M> for ArgsSpy {
        async fn on_tool_call(&self, _: &HookContext, event: ToolCall<'_>) -> ToolCallAction {
            self.0.lock().unwrap().push(event.args.into());
            ToolCallAction::run()
        }
    }

    struct OnToolCallOnly(Arc<AtomicUsize>);
    impl AgentHook<M> for OnToolCallOnly {
        async fn on_tool_call(&self, _: &HookContext, _: ToolCall<'_>) -> ToolCallAction {
            self.0.fetch_add(1, Ordering::Relaxed);
            ToolCallAction::skip("called")
        }
    }

    struct YieldingRewriteFromCallId;
    impl AgentHook<M> for YieldingRewriteFromCallId {
        async fn on_tool_call(&self, _: &HookContext, event: ToolCall<'_>) -> ToolCallAction {
            tokio::task::yield_now().await;
            ToolCallAction::rewrite(json!({"call_id": event.internal_call_id}))
        }
    }

    struct YieldingSkip;
    impl AgentHook<M> for YieldingSkip {
        async fn on_tool_call(&self, _: &HookContext, _: ToolCall<'_>) -> ToolCallAction {
            tokio::task::yield_now().await;
            ToolCallAction::skip("denied")
        }
    }

    async fn resolve(stack: &HookStack<M>) -> (ToolCallAction, Option<Value>) {
        stack.resolve_tool_call(&ctx(), tool_call_event()).await
    }

    #[tokio::test]
    async fn erased_dispatch_uses_the_public_on_tool_call_method() {
        let calls = Arc::new(AtomicUsize::new(0));
        let stack = HookStack::<M>::with(OnToolCallOnly(calls.clone()));

        let (action, salvaged) = resolve(&stack).await;

        assert_eq!(action, ToolCallAction::skip("called"));
        assert_eq!(salvaged, None);
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn string_rewrite_is_json_encoded_for_later_hook_in_same_stack() {
        let spy = ArgsSpy::default();
        let replacement = Value::String("sanitized".into());
        let mut stack = HookStack::<M>::new();
        stack.push(RewriteHook(replacement.clone()));
        stack.push(spy.clone());

        let (action, salvaged) = resolve(&stack).await;

        assert_eq!(action, ToolCallAction::rewrite(replacement.clone()));
        assert_eq!(salvaged, None);
        assert_eq!(
            spy.0.lock().unwrap().as_slice(),
            [serde_json::to_string(&replacement).unwrap()]
        );
    }

    #[tokio::test]
    async fn string_rewrite_is_json_encoded_for_hook_in_nested_stack() {
        let spy = ArgsSpy::default();
        let replacement = Value::String("sanitized".into());
        let inner = HookStack::<M>::with(spy.clone());
        let mut outer = HookStack::<M>::new();
        outer.push(RewriteHook(replacement.clone()));
        outer.push(inner);

        let (action, salvaged) = resolve(&outer).await;

        assert_eq!(action, ToolCallAction::rewrite(replacement.clone()));
        assert_eq!(salvaged, None);
        assert_eq!(
            spy.0.lock().unwrap().as_slice(),
            [serde_json::to_string(&replacement).unwrap()]
        );
    }

    #[tokio::test]
    async fn nested_rewrite_then_skip_preserves_rewrite() {
        let mut inner = HookStack::<M>::new();
        inner.push(RewriteHook(json!({"x":41})));
        inner.push(SkipHook);
        let mut outer = HookStack::<M>::new();
        outer.push(inner);
        let (action, salvaged) = resolve(&outer).await;
        assert!(matches!(action, ToolCallAction::Skip(_)));
        assert_eq!(salvaged, Some(json!({"x":41})));
    }

    #[tokio::test]
    async fn nested_rewrite_then_stop_preserves_rewrite() {
        let mut inner = HookStack::<M>::new();
        inner.push(RewriteHook(json!({"x":41})));
        inner.push(StopHook);
        let mut outer = HookStack::<M>::new();
        outer.push(inner);
        let (action, salvaged) = resolve(&outer).await;
        assert!(matches!(action, ToolCallAction::Stop(_)));
        assert_eq!(salvaged, Some(json!({"x":41})));
    }

    #[tokio::test]
    async fn deeply_nested_terminal_action_preserves_the_last_rewrite() {
        let mut inner = HookStack::<M>::new();
        inner.push(RewriteHook(json!({"x":3})));
        inner.push(SkipHook);

        let mut middle = HookStack::<M>::new();
        middle.push(RewriteHook(json!({"x":2})));
        middle.push(inner);

        let mut outer = HookStack::<M>::new();
        outer.push(RewriteHook(json!({"x":1})));
        outer.push(middle);

        let (action, salvaged) = resolve(&outer).await;

        assert_eq!(action, ToolCallAction::skip("denied"));
        assert_eq!(salvaged, Some(json!({"x":3})));
    }

    #[tokio::test]
    async fn concurrent_nested_resolutions_keep_rewrites_isolated_by_call() {
        let mut inner = HookStack::<M>::new();
        inner.push(YieldingRewriteFromCallId);
        inner.push(YieldingSkip);
        let outer = HookStack::<M>::with(inner);
        let context = ctx();

        let first = outer.resolve_tool_call(
            &context,
            ToolCall {
                internal_call_id: "first",
                ..tool_call_event()
            },
        );
        let second = outer.resolve_tool_call(
            &context,
            ToolCall {
                internal_call_id: "second",
                ..tool_call_event()
            },
        );
        let ((first_action, first_rewrite), (second_action, second_rewrite)) =
            tokio::join!(first, second);

        assert_eq!(first_action, ToolCallAction::skip("denied"));
        assert_eq!(first_rewrite, Some(json!({"call_id": "first"})));
        assert_eq!(second_action, ToolCallAction::skip("denied"));
        assert_eq!(second_rewrite, Some(json!({"call_id": "second"})));
    }

    #[tokio::test]
    async fn outer_rewrite_threads_into_nested_stack() {
        let spy = ArgsSpy::default();
        let mut inner = HookStack::<M>::new();
        inner.push(spy.clone());
        inner.push(SkipHook);
        let mut outer = HookStack::<M>::new();
        outer.push(RewriteHook(json!({"x":1})));
        outer.push(inner);
        let (action, salvaged) = resolve(&outer).await;
        assert!(matches!(action, ToolCallAction::Skip(_)));
        assert_eq!(salvaged, Some(json!({"x":1})));
        assert_eq!(
            spy.0.lock().unwrap().as_slice(),
            [serde_json::to_string(&json!({"x":1})).unwrap()]
        );
    }

    #[tokio::test]
    async fn nested_proceeding_rewrite_surfaces_as_rewrite_action() {
        let mut proceed = HookStack::<M>::new();
        proceed.push(RewriteHook(json!({"x":5})));
        let (action, salvaged) = resolve(&proceed).await;
        assert_eq!(action, ToolCallAction::rewrite(json!({"x":5})));
        assert_eq!(salvaged, None);
    }

    #[test]
    fn action_types_are_event_specific() {
        fn completion(_: CompletionCallAction) {}
        fn call(_: ToolCallAction) {}
        fn result(_: ToolResultAction) {}
        fn invalid(_: InvalidToolCallAction) {}
        fn observation(_: ObservationAction) {}
        completion(CompletionCallAction::continue_run());
        call(ToolCallAction::run());
        result(ToolResultAction::keep());
        invalid(InvalidToolCallAction::fail());
        observation(ObservationAction::continue_run());
        let calls = AtomicUsize::new(0);
        calls.fetch_add(1, Ordering::Relaxed);
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }
}
