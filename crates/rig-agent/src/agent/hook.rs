//! Event-specific hooks for observing and steering an agent run.
//!
//! [`AgentHook`] supplies typed lifecycle actions; [`HookStack`] composes them in
//! registration order. Blocking and streaming runs share lifecycle decisions;
//! streamed deltas remain provisional until the model turn is accepted.
//!
//! ```
//! use rig_agent::agent::{AgentHook, HookContext, OutcomeAction, OutcomeEvent};
//!
//! struct ResponseLogger;
//!
//! impl AgentHook for ResponseLogger {
//!     async fn on_outcome(&self, _ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
//!         if let Some(response) = event.completion() {
//!             println!(
//!                 "message {:?}: {:?} ({:?})",
//!                 response.message_id, response.choice, response.usage
//!             );
//!         }
//!         OutcomeAction::proceed()
//!     }
//! }
//! ```

use rig_core::streaming::BlockId;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{collections::HashMap, future::Future, sync::Arc};

use rig_core::tool::context::TypeMap;
use rig_core::{
    completion::FinishReason,
    message::{AssistantContent, Message},
    wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync},
};

use rig_core::{
    completion::ModelRef,
    effect::{EffectFamily, EffectId, EffectKind, Outcome},
    error::{ErrorKind, ErrorReport},
};

use crate::{
    completion::{ResponseIdentity, Usage},
    json_utils,
    tool::{ToolContext, ToolOutput, ToolResult},
};

pub use rig_core::id::RunId;

pub use crate::run::RunEntry;

/// Run-scoped typed storage shared by hooks. Clones share in-process state;
/// entries are neither serialized nor rewound when a run is forked.
/// Use [`HookContext::append_entry`] for state that must travel with the run.
#[derive(Clone, Default)]
pub struct Scratchpad {
    inner: Arc<std::sync::Mutex<TypeMap>>,
}

impl Scratchpad {
    fn lock(&self) -> std::sync::MutexGuard<'_, TypeMap> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
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

    /// Remove a value or create its default, update it outside the lock, then
    /// insert it and return the closure's result. Reentrant access is allowed;
    /// concurrent updates of the same type are last-writer-wins, not serialized.
    pub fn update<T, R>(&self, update: impl FnOnce(&mut T) -> R) -> R
    where
        T: Clone + Default + WasmCompatSend + WasmCompatSync + 'static,
    {
        let mut value = self.lock().remove::<T>().unwrap_or_default();
        let result = update(&mut value);
        self.lock().insert(value);
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

/// Run-scoped context supplied to hooks.
#[derive(Debug)]
pub struct HookContext {
    /// The bus the run dispatches through, when it has one.
    dispatcher: Option<crate::bus::Dispatcher>,
    run_id: RunId,
    turn: AtomicUsize,
    is_streaming: bool,
    agent_name: Option<String>,
    scratchpad: Scratchpad,
    /// A patch a stack accumulated before one of its hooks denied the
    /// dispatch, keyed by the effect's id. `DispatchAction::Deny` carries only
    /// the report, so the engine reads the effective effect (the arguments a
    /// skipped tool result reports) from here; a nested stack records into
    /// the same slot, so the salvage survives any nesting depth.
    salvaged_patches: std::sync::Mutex<HashMap<EffectId, EffectKind>>,
    /// Seeded and newly appended entries, in append order.
    entries: std::sync::Mutex<Vec<RunEntry>>,
    /// Appends not yet flushed into the [`AgentRun`](crate::agent::AgentRun)
    /// by the driver.
    pending_entries: std::sync::Mutex<Vec<RunEntry>>,
}

impl HookContext {
    pub(crate) fn new(
        is_streaming: bool,
        agent_name: Option<String>,
        dispatcher: Option<crate::bus::Dispatcher>,
    ) -> Self {
        Self {
            dispatcher,
            run_id: RunId::new(),
            turn: AtomicUsize::new(0),
            is_streaming,
            agent_name,
            scratchpad: Scratchpad::default(),
            salvaged_patches: std::sync::Mutex::new(HashMap::new()),
            entries: std::sync::Mutex::new(Vec::new()),
            pending_entries: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Record the patch a stack had accumulated when one of its hooks denied
    /// the dispatch `id`. The innermost stack records first and wins: an
    /// enclosing stack's earlier patch was already threaded into what the
    /// inner stack saw, so the inner accumulation is the last rewrite before
    /// the terminal action.
    fn salvage_patch(&self, id: EffectId, kind: EffectKind) {
        self.salvaged_patches
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .entry(id)
            .or_insert(kind);
    }

    /// The effect as patched before the dispatch `id` was denied, if any.
    pub(crate) fn take_salvaged_patch(&self, id: EffectId) -> Option<EffectKind> {
        self.salvaged_patches
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&id)
    }

    /// Seed the entries a resumed run carried; called by the driver at run
    /// start, before any hook fires.
    pub(crate) fn seed_entries(&self, entries: &[RunEntry]) {
        self.entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .extend_from_slice(entries);
    }

    /// Drain the appends not yet flushed into the run; called by the driver
    /// at each step boundary.
    pub(crate) fn drain_pending_entries(&self) -> Vec<RunEntry> {
        std::mem::take(
            &mut *self
                .pending_entries
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        )
    }

    pub(crate) fn set_turn(&self, turn: usize) {
        self.turn.store(turn, Ordering::Relaxed);
    }

    /// The run's bus, for a hook binding a typed view; fails when the
    /// context was built outside a run.
    fn bus(&self) -> Result<&crate::bus::Dispatcher, ErrorReport> {
        self.dispatcher.as_ref().ok_or_else(|| {
            ErrorReport::new(
                ErrorKind::BusClosed,
                "this hook context was built outside a run and has no bus",
            )
        })
    }

    /// Bind a typed key on the run's bus, borrowing this context. Fails without
    /// a bus or when the current registration has another family. Dispatches are
    /// served and recorded without re-entering hooks; see [`RunHandle`] for
    /// request ownership and driver requirements.
    #[track_caller]
    pub fn bind<'ctx, F: rig_core::effect::Family>(
        &'ctx self,
        key: &rig_core::effect::Key<F>,
    ) -> Result<RunHandle<'ctx, F>, ErrorReport> {
        self.bus()?.bind(key).map(RunHandle::scoped)
    }

    /// Bind the retrieval index under `key` on the run's bus, for a hook
    /// that retrieves for itself; see [`bind`](Self::bind) for the scope.
    /// Fails when the context was built outside a run or `key` serves
    /// another family.
    pub fn index<'ctx>(
        &'ctx self,
        key: &rig_core::effect::HandlerKey,
    ) -> Result<RunHandle<'ctx, rig_core::effect::family::Retrieve>, ErrorReport> {
        self.bus()?.handle(key).map(RunHandle::scoped)
    }

    /// [`index`](Self::index) for a completion model.
    pub fn model<'ctx>(
        &'ctx self,
        key: &rig_core::effect::HandlerKey,
    ) -> Result<RunHandle<'ctx, rig_core::effect::family::Completion>, ErrorReport> {
        self.bus()?.handle(key).map(RunHandle::scoped)
    }

    /// [`index`](Self::index) for a tool.
    pub fn tool<'ctx>(
        &'ctx self,
        key: &rig_core::effect::HandlerKey,
    ) -> Result<RunHandle<'ctx, rig_core::effect::family::Tool>, ErrorReport> {
        self.bus()?.handle(key).map(RunHandle::scoped)
    }

    /// [`index`](Self::index) for conversation memory.
    pub fn memory<'ctx>(
        &'ctx self,
        key: &rig_core::effect::HandlerKey,
    ) -> Result<RunHandle<'ctx, rig_core::effect::family::Memory>, ErrorReport> {
        self.bus()?.handle(key).map(RunHandle::scoped)
    }

    /// Stable run identifier.
    pub fn run_id(&self) -> RunId {
        self.run_id
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

    /// Serialize and append an entry stamped with the current turn. Serialization
    /// errors leave entries unchanged. The driver transfers pending entries into
    /// the serializable run at its next step boundary; this is not a disk write.
    /// Entries appended during `on_run_settled` remain visible locally but are not
    /// transferred into the finished run.
    pub fn append_entry<T: serde::Serialize>(
        &self,
        kind: impl Into<String>,
        value: &T,
    ) -> Result<(), serde_json::Error> {
        let entry = RunEntry {
            kind: kind.into(),
            turn: self.turn(),
            value: serde_json::to_value(value)?,
        };
        self.entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(entry.clone());
        self.pending_entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(entry);
        Ok(())
    }

    /// Clone all entries of `kind`, including resumed entries and new appends,
    /// in append order. Reading does not consume entries.
    pub fn entries(&self, kind: &str) -> Vec<RunEntry> {
        self.entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .filter(|entry| entry.kind == kind)
            .cloned()
            .collect()
    }

    /// Clone the most recent entry of `kind`, if present.
    pub fn last_entry(&self, kind: &str) -> Option<RunEntry> {
        self.entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .rev()
            .find(|entry| entry.kind == kind)
            .cloned()
    }
}

pub use crate::run::policy::{
    InvalidToolCallAction, InvalidToolCallContext, InvalidToolCallReason, RetryRequest,
};

/// Completion-call event.
///
/// Per `CallModel` step, hook resolution is ordered: completion-call hooks run
/// **first** and their [`RequestPatch`]es merge in registration order. Only
/// when every completion-call hook proceeds does [`ModelSelection`] run
/// (receiving the merged patch), after which request preparation inspects the
/// selected model's captured
/// [`ProviderCapabilities`](crate::completion::ProviderCapabilities) and the
/// attempt is issued. A completion-call stop therefore suppresses model
/// selection entirely and does not advance
/// [`ModelSelection::previous_model`].
#[derive(Clone, Copy)]
pub struct CompletionCallEvent<'a> {
    /// Prompt for this turn.
    pub prompt: &'a Message,
    /// History preceding the prompt.
    pub history: &'a [Message],
    /// One-based model-call index.
    pub turn: usize,
}

/// Model-selection event after completion-call hooks proceed and before request
/// preparation inspects the selected model's capabilities. The runner default is
/// the initial candidate on every attempt, including retries; earlier selections
/// in a stack update the candidate seen by later hooks.
///
/// Selection must be synchronous and non-blocking. Prepared attempts retain their
/// chosen handle. `previous_model` advances only when an operation is invoked,
/// including operations that fail, not on hook stops or preparation errors.
#[derive(Clone, Copy)]
pub struct ModelSelection<'a> {
    /// Prompt for the pending model call.
    pub prompt: &'a Message,
    /// Canonical history visible to the pending model call.
    pub history: &'a [Message],
    /// Merged per-turn request patch from this step's completion-call hooks
    /// (in hook registration order), when any hook patched the request.
    pub request_patch: Option<&'a RequestPatch>,
    /// Model that executed the preceding issued attempt in this run, if any.
    pub previous_model: Option<&'a ModelRef>,
    /// Runner default used as the initial candidate for this call.
    pub default_model: &'a ModelRef,
    /// Candidate after all earlier model-selection hooks.
    pub selected_model: &'a ModelRef,
}

impl<'a> ModelSelection<'a> {
    /// Construct a model-selection event from borrowed request and candidate state.
    pub fn new(
        prompt: &'a Message,
        history: &'a [Message],
        request_patch: Option<&'a RequestPatch>,
        previous_model: Option<&'a ModelRef>,
        default_model: &'a ModelRef,
        selected_model: &'a ModelRef,
    ) -> Self {
        Self {
            prompt,
            history,
            request_patch,
            previous_model,
            default_model,
            selected_model,
        }
    }
}

/// Medium-neutral accepted model-turn event.
///
/// The turn is canonicalized and parked in the run state, but has not yet been
/// advanced into tool execution or finalization. A hook may therefore reject a
/// tool-free turn with [`ModelTurnAction::Retry`].
#[derive(Clone, Copy)]
pub struct ModelTurnFinished<'a> {
    /// One-based model-call index.
    pub turn: usize,
    /// Canonical assistant content parked for hook acceptance.
    pub content: &'a Vec<AssistantContent>,
    /// Usage reported for the turn.
    pub usage: Usage,
    /// This exact attempt's response identity metadata, the same value the
    /// preceding completion outcome ([`OutcomeEvent`]) carried. On a retry, this is
    /// the retried attempt's own identity, never a previous attempt's.
    pub identity: &'a ResponseIdentity,
    /// This attempt's normalized terminal reason, or `None` when unreported.
    /// Unknown reasons retain their spelling in [`FinishReason::Other`].
    /// A reported `Stop` is reconciled to `ToolCalls` when calls are present.
    pub finish_reason: Option<&'a FinishReason>,
    /// This attempt's output-token cap after agent configuration, request
    /// overrides, and completion-call patches. `None` leaves the cap to the provider.
    pub max_tokens: Option<u64>,
    /// This attempt's decoded provider response, serialized as JSON on both
    /// streaming and unary surfaces. Hand-constructed responses may use `Null`.
    pub raw: &'a serde_json::Value,
}

/// Action for the medium-neutral [`ModelTurnFinished`] event.
///
/// Every retry consumes the run's existing total model-call budget. Rig does
/// not impose a separate response-retry limit; hooks that need one should keep
/// run-scoped state in [`HookContext::scratchpad`]. Retrying a turn containing
/// tool calls is rejected so provider-visible history never contains unanswered
/// calls. Use tool-call hooks to steer those turns instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelTurnAction {
    /// Accept the turn and continue the run.
    Continue,
    /// Reject the turn and request another model call.
    Retry(RetryRequest),
    /// Stop the run with a reason.
    Stop(String),
}

impl ModelTurnAction {
    /// Accepts the completed model turn.
    pub fn continue_run() -> Self {
        Self::Continue
    }

    /// Discards the response and reuses the same prompt and preceding history
    /// with fresh request preparation.
    pub fn repeat() -> Self {
        Self::Retry(RetryRequest::Repeat)
    }

    /// Preserves the response, appends corrective feedback, and retries.
    pub fn retry_with_feedback(feedback: impl Into<String>) -> Self {
        Self::Retry(RetryRequest::Feedback(feedback.into()))
    }

    /// Stops the run with the supplied reason.
    pub fn stop(reason: impl Into<String>) -> Self {
        Self::Stop(reason.into())
    }
}

/// Streaming text delta.
#[derive(Clone, Copy)]
pub struct TextDelta<'a> {
    /// Newly received text.
    pub delta: &'a str,
    /// Text accumulated for the turn.
    pub aggregated: &'a str,
}

/// Streaming reasoning delta.
#[derive(Clone, Copy)]
pub struct ReasoningDelta<'a> {
    /// The stream block this reasoning part streams under: stable across
    /// the part's deltas and its eventual completed reasoning item. A minted
    /// block id is never persisted as a provider-issued reasoning id.
    pub id: &'a BlockId,
    /// Provider-issued durable reasoning item id, when the wire provides one.
    pub provider_id: Option<&'a str>,
    /// Newly received reasoning fragment.
    pub delta: &'a str,
    /// Reasoning text accumulated for this reasoning part through this delta.
    pub aggregated: &'a str,
}

/// Streaming tool-call delta.
#[derive(Clone, Copy)]
pub struct ToolCallDelta<'a> {
    /// Stable block ID shared by fragments and the completed call's
    /// [`DispatchEvent`]. Provider-issued IDs arrive with the completed call.
    pub block_id: &'a BlockId,
    /// Tool name on the first delta.
    pub tool_name: Option<&'a str>,
    /// Newly received argument fragment.
    pub delta: &'a str,
}

/// Initial prompt event before the first completion-call hook. In a [`HookStack`],
/// rewrites reach later hooks in registration order; the first stop prevents
/// remaining hooks and provider calls.
#[derive(Clone, Copy)]
pub struct RunStart<'a> {
    /// The prompt the run will send on its first model call, including
    /// earlier hooks' rewrites.
    pub prompt: &'a Message,
    /// The input chat history preceding the prompt.
    pub history: &'a [Message],
}

/// Action for the pre-run [`RunStart`] event.
#[derive(Debug, Clone, PartialEq)]
pub enum RunStartAction {
    /// Start the run with the current prompt.
    Continue,
    /// Replace the prompt and pass it to later hooks.
    Rewrite(Message),
    /// Stop the run before any model call, with a reason.
    Stop(String),
}

impl RunStartAction {
    /// Starts the run with the current prompt.
    pub fn continue_run() -> Self {
        Self::Continue
    }

    /// Replaces the prompt; later hooks in a [`HookStack`] see the rewrite.
    pub fn rewrite(prompt: impl Into<Message>) -> Self {
        Self::Rewrite(prompt.into())
    }

    /// Stops the run before any provider call.
    pub fn stop(reason: impl Into<String>) -> Self {
        Self::Stop(reason.into())
    }
}

/// Terminal run event carrying a final response or error. No retry, model call,
/// or tool execution follows automatically.
#[derive(Clone, Copy)]
pub struct RunSettled<'a> {
    /// How the run ended.
    pub outcome: SettledOutcome<'a>,
    /// Messages actually committed by this run, including its prompt and
    /// excluding input history. Available on error as well as success;
    /// rejected or truncated answerless turns are not committed.
    /// `None` means the run failed before its state was constructed.
    pub messages: Option<&'a [Message]>,
}

/// The outcome carried by [`RunSettled`].
#[derive(Clone, Copy)]
pub enum SettledOutcome<'a> {
    /// The run completed with this final response.
    Response(&'a crate::run::response::PromptResponse),
    /// The run terminated with an error, rendered via its `Display` form.
    Error(&'a str),
}

/// Hook event kind used only as an observation performance hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StepEventKind {
    /// `on_run_start`.
    RunStart,
    /// `on_run_settled`.
    RunSettled,
    /// `on_completion_call`.
    CompletionCall,
    /// `on_model_turn_finished`.
    ModelTurnFinished,
    /// `on_invalid_tool_call`.
    InvalidToolCall,
    /// `on_text_delta`.
    TextDelta,
    /// `on_reasoning_delta`.
    ReasoningDelta,
    /// `on_tool_call_delta`.
    ToolCallDelta,
    /// `on_dispatch`/`on_outcome` for a completion effect.
    CompletionDispatch,
    /// `on_dispatch`/`on_outcome` for a tool-call effect.
    ToolDispatch,
    /// `on_dispatch`/`on_outcome` for an embedding effect (observe-only by
    /// default: opt in through `observes`).
    EmbedDispatch,
    /// `on_dispatch`/`on_outcome` for a reranking effect (observe-only by
    /// default).
    RerankDispatch,
    /// `on_dispatch`/`on_outcome` for a conversation-memory effect
    /// (observe-only by default).
    MemoryDispatch,
    /// `on_dispatch`/`on_outcome` for a retrieval effect (observe-only by
    /// default).
    RetrieveDispatch,
    /// `on_dispatch`/`on_outcome` for a custom effect (observe-only by
    /// default).
    CustomDispatch,
}

impl StepEventKind {
    /// The dispatch-boundary event kind for an effect family.
    pub const fn for_family(family: EffectFamily) -> Self {
        match family {
            EffectFamily::Completion => Self::CompletionDispatch,
            EffectFamily::Tool => Self::ToolDispatch,
            EffectFamily::Embed => Self::EmbedDispatch,
            EffectFamily::Rerank => Self::RerankDispatch,
            EffectFamily::Memory => Self::MemoryDispatch,
            EffectFamily::Retrieve => Self::RetrieveDispatch,
            EffectFamily::Custom => Self::CustomDispatch,
        }
    }
}

/// An effect about to be dispatched: what `on_dispatch` sees.
#[derive(Clone, Copy)]
pub struct DispatchEvent<'a> {
    /// The dispatch's id, minted before the hook runs so an observation can
    /// be correlated with the bus-tap record.
    pub id: EffectId,
    /// The effect, after any earlier hook's patch.
    pub kind: &'a EffectKind,
    /// The turn the effect belongs to.
    pub turn: usize,
    /// The block the effect answers, for a tool call the model emitted.
    pub block_id: Option<&'a BlockId>,
    /// Tool invocation context, carried separately from the effect.
    pub context: Option<&'a ToolContext>,
}

/// Dispatch-boundary action.
#[derive(Debug, Clone)]
pub enum DispatchAction {
    /// Dispatch as is.
    Proceed,
    /// Dispatch this effect instead. A patch must keep the family; the
    /// stack rejects a family change as an internal error before later hooks
    /// observe it. Tool calls must also retain their target name; only their
    /// arguments may be patched.
    Patch(EffectKind),
    /// Do not dispatch: the effect resolves failed with this report and
    /// never reaches a handler. For a tool call a report of kind
    /// `Cancelled` cancels the run; any other kind becomes the skipped
    /// result the model sees. For a completion any report fails the turn.
    Deny(ErrorReport),
}

/// Validate before a patch becomes visible to policy or execution.
pub(crate) fn validate_dispatch_patch(
    current: &EffectKind,
    next: &EffectKind,
) -> Result<(), ErrorReport> {
    if current.family() != next.family() {
        return Err(ErrorReport::new(
            ErrorKind::Internal,
            format!(
                "a hook patched a {} dispatch into a `{}` effect",
                current.name(),
                next.name()
            ),
        ));
    }
    if let (EffectKind::ToolCall { name: current, .. }, EffectKind::ToolCall { name: next, .. }) =
        (current, next)
        && current != next
    {
        return Err(ErrorReport::new(
            ErrorKind::Other,
            "a dispatch patch cannot change the tool target",
        ));
    }
    Ok(())
}

impl DispatchAction {
    /// Dispatch as is.
    pub fn proceed() -> Self {
        Self::Proceed
    }

    /// Dispatch this effect instead.
    pub fn patch(kind: EffectKind) -> Self {
        Self::Patch(kind)
    }

    /// Deny with a report.
    pub fn deny(report: ErrorReport) -> Self {
        Self::Deny(report)
    }

    /// Deny a tool call so the model sees it as skipped with `reason`.
    pub fn skip(reason: impl Into<String>) -> Self {
        Self::Deny(ErrorReport::new(ErrorKind::Other, reason))
    }

    /// Deny and cancel the run with `reason`.
    pub fn stop(reason: impl Into<String>) -> Self {
        Self::Deny(ErrorReport::new(ErrorKind::Cancelled, reason))
    }

    /// Patch a tool call's arguments, keeping its name and context.
    /// `Proceed` when `kind` is not a tool call.
    pub fn rewrite_tool_args(kind: &EffectKind, args: impl Into<serde_json::Value>) -> Self {
        match kind {
            EffectKind::ToolCall { name, .. } => Self::Patch(EffectKind::ToolCall {
                name: name.clone(),
                args: json_utils::serialize_json_value(&args.into()),
            }),
            _ => Self::Proceed,
        }
    }

    /// Serialize replacement arguments and patch the tool call with them.
    pub fn try_rewrite_tool_args<T: serde::Serialize>(
        kind: &EffectKind,
        args: &T,
    ) -> Result<Self, serde_json::Error> {
        Ok(Self::rewrite_tool_args(kind, serde_json::to_value(args)?))
    }
}

impl<'a> DispatchEvent<'a> {
    /// The tool name, for a tool-call effect.
    pub fn tool_name(&self) -> Option<&'a str> {
        match self.kind {
            EffectKind::ToolCall { name, .. } => Some(name),
            _ => None,
        }
    }

    /// The JSON arguments (after earlier patches), for a tool-call effect.
    pub fn tool_args(&self) -> Option<&'a str> {
        match self.kind {
            EffectKind::ToolCall { args, .. } => Some(args),
            _ => None,
        }
    }

    /// The dispatch context, for a tool-call effect.
    pub fn tool_context(&self) -> Option<&'a ToolContext> {
        self.context
    }

    /// The request about to be sent, for a completion effect.
    pub fn completion_request(&self) -> Option<&'a rig_core::completion::CompletionRequest> {
        match self.kind {
            EffectKind::Completion { request, .. } => Some(request),
            _ => None,
        }
    }
}

/// An effect's answer: what `on_outcome` sees.
#[derive(Clone, Copy)]
pub struct OutcomeEvent<'a> {
    /// The dispatch's id.
    pub id: EffectId,
    /// The effect that was dispatched (after patches).
    pub kind: &'a EffectKind,
    /// The answer, after any earlier hook's replacement.
    pub outcome: &'a Result<Outcome, ErrorReport>,
    /// The turn the effect belongs to.
    pub turn: usize,
    /// The block the effect answered, for a tool call the model emitted.
    pub block_id: Option<&'a BlockId>,
    /// Published tool context, carried separately from the outcome.
    pub context: Option<&'a ToolContext>,
}

/// Outcome-boundary action.
#[derive(Debug, Clone)]
pub enum OutcomeAction {
    /// Keep the answer.
    Proceed,
    /// Use this answer instead.
    Replace(Result<Outcome, ErrorReport>),
}

impl OutcomeAction {
    /// Keep the answer.
    pub fn proceed() -> Self {
        Self::Proceed
    }

    /// Use this answer instead.
    pub fn replace(outcome: Result<Outcome, ErrorReport>) -> Self {
        Self::Replace(outcome)
    }

    /// Stop the run with `reason`: a replacement whose error is `Cancelled`
    /// terminates the run instead of being delivered. This is how a hook
    /// that observed an answer (a completion, a tool result) ends the run.
    /// The cancellation short-circuits nested hook stacks; later hooks cannot
    /// replace it with a successful answer.
    pub fn stop(reason: impl Into<String>) -> Self {
        Self::Replace(Err(ErrorReport::new(ErrorKind::Cancelled, reason)))
    }

    /// Replace the model-visible output of a tool result, keeping the
    /// result's status and the dispatch context. `Proceed` when `event`
    /// did not resolve to a tool result.
    pub fn rewrite_tool_output(event: &OutcomeEvent<'_>, output: ToolOutput) -> Self {
        match event.outcome {
            Ok(Outcome::ToolResult { result }) => Self::Replace(Ok(Outcome::ToolResult {
                result: result.clone().with_output(output),
            })),
            _ => Self::Proceed,
        }
    }

    /// [`OutcomeAction::rewrite_tool_output`] with a text output.
    pub fn rewrite_tool_result(event: &OutcomeEvent<'_>, text: impl Into<String>) -> Self {
        Self::rewrite_tool_output(event, ToolOutput::text(text))
    }
}

impl<'a> OutcomeEvent<'a> {
    /// The tool result this outcome carries, for a tool-call effect.
    pub fn tool_result(&self) -> Option<&'a ToolResult> {
        match self.outcome {
            Ok(Outcome::ToolResult { result, .. }) => Some(result),
            _ => None,
        }
    }

    /// The dispatch context the tool answered with, for a tool-call effect.
    pub fn tool_context(&self) -> Option<&'a ToolContext> {
        self.context
    }

    /// The completion this outcome carries, for a completion effect.
    pub fn completion(&self) -> Option<&'a rig_core::completion::CompletionResponse> {
        match self.outcome {
            Ok(Outcome::Completion(response)) => Some(response),
            _ => None,
        }
    }

    /// The tool name, for a tool-call effect.
    pub fn tool_name(&self) -> Option<&'a str> {
        match self.kind {
            EffectKind::ToolCall { name, .. } => Some(name),
            _ => None,
        }
    }

    /// The effective JSON arguments, for a tool-call effect.
    pub fn tool_args(&self) -> Option<&'a str> {
        match self.kind {
            EffectKind::ToolCall { args, .. } => Some(args),
            _ => None,
        }
    }
}

pub use crate::run::patch::RequestPatch;

/// Action for model-selection hooks.
#[derive(Debug, Clone)]
pub enum ModelSelectionAction {
    /// Keep the candidate supplied to this hook.
    Continue,
    /// Replace the candidate and pass it to later hooks: the label of a
    /// model registered on the agent's bus.
    Select(ModelRef),
    /// Stop the run before request preparation or model execution.
    Stop(String),
}

impl ModelSelectionAction {
    /// Keeps the current model candidate.
    pub fn continue_run() -> Self {
        Self::Continue
    }

    /// Selects `model` and passes it to later hooks.
    pub fn select(model: impl Into<ModelRef>) -> Self {
        Self::Select(model.into())
    }

    /// Stops the run before the pending model attempt.
    ///
    /// A selection stop happens before the attempt is issued, so it does not
    /// advance [`ModelSelection::previous_model`].
    pub fn stop(reason: impl Into<String>) -> Self {
        Self::Stop(reason.into())
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
pub trait AgentHook: WasmCompatSend + WasmCompatSync {
    /// Replay identity recorded by the hook stack. `None` uses the unqualified
    /// type name; include decision-affecting configuration in an explicit name.
    fn name(&self) -> Option<String> {
        None
    }

    /// Runs once before the run's first model call, seeing the initial prompt.
    ///
    /// The hook may rewrite the prompt or stop the run before any provider
    /// call. In a [`HookStack`], rewrites chain in registration order and the
    /// first stop wins; see [`RunStart`]. The default action starts the run
    /// with the current prompt.
    fn on_run_start(
        &self,
        _ctx: &HookContext,
        _event: RunStart<'_>,
    ) -> impl Future<Output = RunStartAction> + WasmCompatSend {
        async { RunStartAction::Continue }
    }

    /// Observe the final response or terminal error without scheduling more work.
    /// Existing entries are visible, but entries appended here are not transferred
    /// into the finished run.
    fn on_run_settled(
        &self,
        _ctx: &HookContext,
        _event: RunSettled<'_>,
    ) -> impl Future<Output = ()> + WasmCompatSend {
        async {}
    }

    /// Selects the model for the pending model-call boundary.
    ///
    /// Selection is synchronous, local, and non-blocking: it operates only on
    /// already-constructed [`ModelRef`] values and may read or write the
    /// run [`Scratchpad`], but must not perform blocking I/O. It runs once per
    /// `CallModel` step whose completion-call hooks proceed, including
    /// retries and post-tool calls, never after a completion-call stop, and
    /// in-flight attempts never rebind. In a [`HookStack`], selections are
    /// passed to later hooks in registration order; the last selection wins
    /// and a stop is terminal. The default action keeps the current candidate.
    /// See [`ModelSelection`] for the full ordering contract.
    fn on_model_select(
        &self,
        _ctx: &HookContext,
        _event: ModelSelection<'_>,
    ) -> ModelSelectionAction {
        ModelSelectionAction::Continue
    }

    /// Runs before a completion request is sent.
    ///
    /// Return a per-turn patch, continue without one, or stop the run. Patches
    /// from a [`HookStack`] are merged in hook registration order.
    fn on_completion_call(
        &self,
        _ctx: &HookContext,
        _event: CompletionCallEvent<'_>,
    ) -> impl Future<Output = CompletionCallAction> + WasmCompatSend {
        async { CompletionCallAction::Continue }
    }

    /// Observes or rejects the content produced at the end of a model turn.
    ///
    /// A retry is valid only for a tool-free turn and consumes the existing
    /// total model-call budget. The default action accepts the turn.
    fn on_model_turn_finished(
        &self,
        _ctx: &HookContext,
        _event: ModelTurnFinished<'_>,
    ) -> impl Future<Output = ModelTurnAction> + WasmCompatSend {
        async { ModelTurnAction::Continue }
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

    /// Observes a reasoning delta from a streaming response.
    ///
    /// The aggregate is scoped to the reasoning part identified by the event's
    /// correlator. Like all streamed deltas, it remains provisional until the
    /// model turn is accepted. The default action continues the run.
    fn on_reasoning_delta(
        &self,
        _ctx: &HookContext,
        _event: ReasoningDelta<'_>,
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

    /// An effect is about to be dispatched on the agent's bus. Runs for
    /// every family; `Memory`, `Retrieve`, `Embed`, `Rerank` and `Custom` dispatches
    /// are observe-only unless the hook opts in through
    /// [`AgentHook::observes`] for their [`StepEventKind`].
    ///
    /// Gated by [`AgentHook::observes`]: a hook whose `observes` answers
    /// `false` for [`StepEventKind::CompletionDispatch`] or
    /// [`StepEventKind::ToolDispatch`] is **not called** for those
    /// dispatches and cannot deny, patch or replace them. A hook that
    /// overrides `observes` to trim delta noise must keep the dispatch
    /// kinds it means to gate.
    fn on_dispatch(
        &self,
        _ctx: &HookContext,
        _event: DispatchEvent<'_>,
    ) -> impl Future<Output = DispatchAction> + WasmCompatSend {
        async { DispatchAction::Proceed }
    }

    /// An effect resolved on the agent's bus. Gated by
    /// [`AgentHook::observes`] like [`AgentHook::on_dispatch`].
    fn on_outcome(
        &self,
        _ctx: &HookContext,
        _event: OutcomeEvent<'_>,
    ) -> impl Future<Output = OutcomeAction> + WasmCompatSend {
        async { OutcomeAction::Proceed }
    }

    /// Declare event interest. Memory, retrieval, embedding, reranking, and custom
    /// dispatches default to false; all other kinds default to true.
    /// For dispatch events, false prevents both boundary callbacks and therefore
    /// prevents this hook from steering that family.
    fn observes(&self, kind: StepEventKind) -> bool {
        !matches!(
            kind,
            StepEventKind::EmbedDispatch
                | StepEventKind::RerankDispatch
                | StepEventKind::MemoryDispatch
                | StepEventKind::RetrieveDispatch
                | StepEventKind::CustomDispatch
        )
    }
}

impl AgentHook for () {
    fn observes(&self, _kind: StepEventKind) -> bool {
        false
    }
}

/// The erased hook events whose dispatch is a plain `Box::pin(self.on_*(..))`.
/// `model_select` (sync) and `invalid_tool_call` (borrowed event) are
/// hand-written below.
macro_rules! for_each_boxed_hook_event {
    ($m:ident) => {
        $m!(
            completion_call,
            on_completion_call,
            CompletionCallEvent,
            CompletionCallAction
        );
        $m!(
            model_turn_finished,
            on_model_turn_finished,
            ModelTurnFinished,
            ModelTurnAction
        );
        $m!(text_delta, on_text_delta, TextDelta, ObservationAction);
        $m!(
            reasoning_delta,
            on_reasoning_delta,
            ReasoningDelta,
            ObservationAction
        );
        $m!(
            tool_call_delta,
            on_tool_call_delta,
            ToolCallDelta,
            ObservationAction
        );
    };
}

macro_rules! erased_hook_decl {
    ($erased:ident, $on:ident, $event:ident, $action:ident) => {
        fn $erased<'a>(
            &'a self,
            ctx: &'a HookContext,
            event: $event<'a>,
        ) -> WasmBoxedFuture<'a, $action>;
    };
}

macro_rules! erased_hook_forward {
    ($erased:ident, $on:ident, $event:ident, $action:ident) => {
        fn $erased<'a>(
            &'a self,
            ctx: &'a HookContext,
            event: $event<'a>,
        ) -> WasmBoxedFuture<'a, $action> {
            Box::pin(self.$on(ctx, event))
        }
    };
}

trait DynAgentHook: WasmCompatSend + WasmCompatSync {
    fn run_start<'a>(
        &'a self,
        ctx: &'a HookContext,
        event: RunStart<'a>,
    ) -> WasmBoxedFuture<'a, RunStartAction>;
    fn run_settled<'a>(
        &'a self,
        ctx: &'a HookContext,
        event: RunSettled<'a>,
    ) -> WasmBoxedFuture<'a, ()>;
    fn model_select(&self, ctx: &HookContext, event: ModelSelection<'_>) -> ModelSelectionAction;
    fn invalid_tool_call<'a>(
        &'a self,
        ctx: &'a HookContext,
        event: &'a InvalidToolCallContext,
    ) -> WasmBoxedFuture<'a, Option<InvalidToolCallAction>>;
    fn dispatch<'a>(
        &'a self,
        ctx: &'a HookContext,
        event: DispatchEvent<'a>,
    ) -> WasmBoxedFuture<'a, DispatchAction>;
    fn outcome<'a>(
        &'a self,
        ctx: &'a HookContext,
        event: OutcomeEvent<'a>,
    ) -> WasmBoxedFuture<'a, OutcomeAction>;
    for_each_boxed_hook_event!(erased_hook_decl);
    fn observes(&self, kind: StepEventKind) -> bool;
}

impl<H> DynAgentHook for H
where
    H: AgentHook,
{
    fn run_start<'a>(
        &'a self,
        ctx: &'a HookContext,
        event: RunStart<'a>,
    ) -> WasmBoxedFuture<'a, RunStartAction> {
        Box::pin(self.on_run_start(ctx, event))
    }

    fn run_settled<'a>(
        &'a self,
        ctx: &'a HookContext,
        event: RunSettled<'a>,
    ) -> WasmBoxedFuture<'a, ()> {
        Box::pin(self.on_run_settled(ctx, event))
    }

    fn model_select(&self, ctx: &HookContext, event: ModelSelection<'_>) -> ModelSelectionAction {
        self.on_model_select(ctx, event)
    }

    fn invalid_tool_call<'a>(
        &'a self,
        ctx: &'a HookContext,
        event: &'a InvalidToolCallContext,
    ) -> WasmBoxedFuture<'a, Option<InvalidToolCallAction>> {
        Box::pin(self.on_invalid_tool_call(ctx, event))
    }
    fn dispatch<'a>(
        &'a self,
        ctx: &'a HookContext,
        event: DispatchEvent<'a>,
    ) -> WasmBoxedFuture<'a, DispatchAction> {
        Box::pin(self.on_dispatch(ctx, event))
    }
    fn outcome<'a>(
        &'a self,
        ctx: &'a HookContext,
        event: OutcomeEvent<'a>,
    ) -> WasmBoxedFuture<'a, OutcomeAction> {
        Box::pin(self.on_outcome(ctx, event))
    }
    for_each_boxed_hook_event!(erased_hook_forward);
    fn observes(&self, kind: StepEventKind) -> bool {
        AgentHook::observes(self, kind)
    }
}

/// Ordered composable hook stack.
///
/// Model selections chain in registration order: each hook sees the candidate
/// selected by earlier hooks, the last selection wins, and a stop is terminal.
/// Nested stacks preserve the same composition semantics.
#[derive(Clone, Default)]
pub struct HookStack {
    hooks: Vec<Arc<dyn DynAgentHook>>,
    /// The name of every hook ([`AgentHook::name`], else its type name),
    /// in registration order, nested stacks flattened: what an effect
    /// log's header records as the program.
    names: Vec<String>,
}

impl std::fmt::Debug for HookStack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HookStack")
            .field("len", &self.hooks.len())
            .finish()
    }
}

/// `H`'s type name without its module path (generic arguments kept), the
/// name a [`HookStack`] records for it.
fn short_type_name<H>() -> &'static str {
    let full = std::any::type_name::<H>();
    let generics = full.find('<').unwrap_or(full.len());
    let path = &full[..generics];
    match path.rfind("::") {
        Some(at) => &full[at + 2..],
        None => full,
    }
}

impl HookStack {
    /// Creates an empty hook stack.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a hook stack containing `hook`.
    pub fn with<H: AgentHook + 'static>(hook: H) -> Self {
        let mut stack = Self::new();
        stack.push(hook);
        stack
    }

    /// Append a hook. Its recorded name is [`AgentHook::name`] or its type name
    /// without the module path; nested stacks contribute flattened names.
    pub fn push<H: AgentHook + 'static>(&mut self, hook: H) {
        // A nested stack contributes its members' names, flattened in
        // order, so two stacks that run the same hooks name the same program.
        match (&hook as &dyn std::any::Any).downcast_ref::<HookStack>() {
            Some(nested) => self.names.extend(nested.names.iter().cloned()),
            None => self.names.push(
                hook.name()
                    .unwrap_or_else(|| short_type_name::<H>().to_owned()),
            ),
        }
        self.hooks.push(Arc::new(hook));
    }

    /// Recorded hook names in registration order, with nested stacks flattened.
    pub fn names(&self) -> Vec<String> {
        self.names.clone()
    }

    /// Returns `true` when the stack contains no hooks.
    pub fn is_empty(&self) -> bool {
        self.hooks.is_empty()
    }

    /// Returns the number of hooks in the stack.
    pub fn len(&self) -> usize {
        self.hooks.len()
    }
}

/// An action with a neutral `Continue` state that observe-only and steering
/// dispatch short-circuits on: the first non-`Continue` action wins and later
/// hooks are not invoked.
trait ShortCircuitAction: Sized {
    const CONTINUE: Self;
    fn is_continue(&self) -> bool;
}

impl ShortCircuitAction for ObservationAction {
    const CONTINUE: Self = ObservationAction::Continue;
    fn is_continue(&self) -> bool {
        matches!(self, ObservationAction::Continue)
    }
}

impl ShortCircuitAction for ModelTurnAction {
    const CONTINUE: Self = ModelTurnAction::Continue;
    fn is_continue(&self) -> bool {
        matches!(self, ModelTurnAction::Continue)
    }
}

/// Dispatches to each hook in registration order, returning the first action
/// that is not `Continue` without invoking the remaining hooks.
async fn first_non_continue<'a, A, F>(hooks: &'a [Arc<dyn DynAgentHook>], mut dispatch: F) -> A
where
    A: ShortCircuitAction,
    F: FnMut(&'a dyn DynAgentHook) -> WasmBoxedFuture<'a, A>,
{
    for hook in hooks {
        let action = dispatch(hook.as_ref()).await;
        if !action.is_continue() {
            return action;
        }
    }
    A::CONTINUE
}

/// Generate the `HookStack` methods whose dispatch is exactly
/// [`first_non_continue`] over the erased hooks: `(on_* name, erased name,
/// event type, action type)`, mirroring `for_each_boxed_hook_event!`. The
/// genuinely chaining events (`on_model_select`, `on_completion_call`,
/// `on_invalid_tool_call`, `on_dispatch`, `on_outcome`) stay hand-written.
macro_rules! stack_first_non_continue {
    ($($on:ident, $erased:ident, $event:ident, $action:ident;)+) => {
        $(
            async fn $on(&self, ctx: &HookContext, event: $event<'_>) -> $action {
                first_non_continue(&self.hooks, |hook| hook.$erased(ctx, event)).await
            }
        )+
    };
}

impl AgentHook for HookStack {
    async fn on_run_start(&self, ctx: &HookContext, event: RunStart<'_>) -> RunStartAction {
        let mut rewritten: Option<Message> = None;
        for hook in &self.hooks {
            let current = RunStart {
                prompt: rewritten.as_ref().unwrap_or(event.prompt),
                ..event
            };
            match hook.run_start(ctx, current).await {
                RunStartAction::Continue => {}
                RunStartAction::Rewrite(prompt) => rewritten = Some(prompt),
                stop @ RunStartAction::Stop(_) => return stop,
            }
        }
        rewritten.map_or(RunStartAction::Continue, RunStartAction::Rewrite)
    }

    async fn on_run_settled(&self, ctx: &HookContext, event: RunSettled<'_>) {
        for hook in &self.hooks {
            hook.run_settled(ctx, event).await;
        }
    }

    fn on_model_select(
        &self,
        ctx: &HookContext,
        event: ModelSelection<'_>,
    ) -> ModelSelectionAction {
        let mut selected = None;
        for hook in &self.hooks {
            let action = {
                let selected_model = selected.as_ref().unwrap_or(event.selected_model);
                hook.model_select(
                    ctx,
                    ModelSelection {
                        selected_model,
                        ..event
                    },
                )
            };
            match action {
                ModelSelectionAction::Continue => {}
                ModelSelectionAction::Select(model) => selected = Some(model),
                stop @ ModelSelectionAction::Stop(_) => return stop,
            }
        }
        selected.map_or(ModelSelectionAction::Continue, ModelSelectionAction::Select)
    }

    async fn on_completion_call(
        &self,
        ctx: &HookContext,
        event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        let mut merged: Option<RequestPatch> = None;
        for hook in &self.hooks {
            match hook.completion_call(ctx, event).await {
                CompletionCallAction::Continue => {}
                CompletionCallAction::Patch(patch) => {
                    merged = Some(match merged {
                        None => patch,
                        Some(value) => value.merge(patch),
                    });
                }
                stop @ CompletionCallAction::Stop(_) => return stop,
            }
        }
        match merged {
            Some(patch) if !patch.is_empty() => CompletionCallAction::Patch(patch),
            _ => CompletionCallAction::Continue,
        }
    }

    stack_first_non_continue! {
        on_model_turn_finished, model_turn_finished, ModelTurnFinished, ModelTurnAction;
        on_text_delta, text_delta, TextDelta, ObservationAction;
        on_reasoning_delta, reasoning_delta, ReasoningDelta, ObservationAction;
        on_tool_call_delta, tool_call_delta, ToolCallDelta, ObservationAction;
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
    async fn on_dispatch(&self, ctx: &HookContext, event: DispatchEvent<'_>) -> DispatchAction {
        let kind = StepEventKind::for_family(event.kind.family());
        let mut patched: Option<EffectKind> = None;
        for hook in &self.hooks {
            if !hook.observes(kind) {
                continue;
            }
            let current = DispatchEvent {
                kind: patched.as_ref().unwrap_or(event.kind),
                ..event
            };
            match hook.dispatch(ctx, current).await {
                DispatchAction::Proceed => {}
                DispatchAction::Patch(next) => {
                    if let Err(report) = validate_dispatch_patch(current.kind, &next) {
                        if let Some(kind) = patched {
                            ctx.salvage_patch(event.id, kind);
                        }
                        return DispatchAction::Deny(report);
                    }
                    patched = Some(next);
                }
                deny @ DispatchAction::Deny(_) => {
                    // The denial wins, but an earlier hook's patch is what
                    // the skipped result must report: keep it for the engine.
                    if let Some(kind) = patched {
                        ctx.salvage_patch(event.id, kind);
                    }
                    return deny;
                }
            }
        }
        patched.map_or(DispatchAction::Proceed, DispatchAction::Patch)
    }

    async fn on_outcome(&self, ctx: &HookContext, event: OutcomeEvent<'_>) -> OutcomeAction {
        let kind = StepEventKind::for_family(event.kind.family());
        let mut replaced: Option<Result<Outcome, ErrorReport>> = None;
        for hook in &self.hooks {
            if !hook.observes(kind) {
                continue;
            }
            let current = OutcomeEvent {
                outcome: replaced.as_ref().unwrap_or(event.outcome),
                ..event
            };
            match hook.outcome(ctx, current).await {
                OutcomeAction::Proceed => {}
                OutcomeAction::Replace(next) => {
                    if matches!(&next, Err(report) if report.kind == ErrorKind::Cancelled) {
                        return OutcomeAction::Replace(next);
                    }
                    replaced = Some(next);
                }
            }
        }
        replaced.map_or(OutcomeAction::Proceed, OutcomeAction::Replace)
    }

    fn observes(&self, kind: StepEventKind) -> bool {
        self.hooks.iter().any(|hook| hook.observes(kind))
    }
}

/// Typed bus view borrowing a hook context. It cannot be stored as `'static`
/// or cloned into an owned handle. Hosts needing an owned handle must bind
/// through their own [`Dispatcher`](crate::bus::Dispatcher).
///
/// The lifetime constrains the view, not the request futures returned by
/// [`dispatch`](Self::dispatch), [`complete`](Self::complete), or
/// [`top_n`](Self::top_n). Those futures own a single dispatch and can outlive
/// the view. They do not retain permission to create further dispatches and
/// do not drive the bus themselves. Await them within the hook while the run's
/// driver is serving, or arrange continued serving through a host-owned driver.
/// Dropping an unfinished request future cancels that dispatch; dropping this
/// view alone does not. A future retained beyond the run is not a promise that
/// the run's driver will continue serving it.
///
/// ```compile_fail
/// use rig_agent::agent::{HookContext, RunHandle};
/// use rig_core::effect::{Key, family};
///
/// fn escape(ctx: &HookContext, key: &Key<family::Completion>) {
///     let handle = ctx.bind(key).unwrap();
///     // `handle` borrows `ctx`; the task must be `'static`.
///     tokio::spawn(async move {
///         let _ = handle.key();
///     });
/// }
/// ```
///
/// ```compile_fail
/// use rig_agent::agent::{HookContext, RunHandle};
/// use rig_core::effect::{Key, family};
///
/// struct Stash {
///     kept: std::sync::Mutex<Option<RunHandle<'static, family::Completion>>>,
/// }
///
/// fn stash(stash: &Stash, ctx: &HookContext, key: &Key<family::Completion>) {
///     *stash.kept.lock().unwrap() = Some(ctx.bind(key).unwrap());
/// }
/// ```
pub struct RunHandle<'ctx, F: rig_core::effect::Family> {
    inner: crate::bus::Handle<F>,
    _run: std::marker::PhantomData<&'ctx HookContext>,
}

impl<'ctx, F: rig_core::effect::Family> RunHandle<'ctx, F> {
    fn scoped(inner: crate::bus::Handle<F>) -> Self {
        Self {
            inner,
            _run: std::marker::PhantomData,
        }
    }

    /// Dispatch a typed request of this family.
    pub fn dispatch(&self, request: F::Request) -> crate::bus::Typed<F> {
        self.inner.dispatch(request)
    }

    /// The key this view dispatches to.
    pub fn key(&self) -> &rig_core::effect::HandlerKey {
        self.inner.key()
    }

    /// The descriptor now (a runtime replacement under the key is visible).
    pub fn descriptor(&self) -> rig_core::effect::HandlerDescriptor {
        self.inner.descriptor()
    }

    /// Whether the bus behind this view has closed.
    pub fn is_closed(&self) -> bool {
        self.inner.is_closed()
    }
}

impl RunHandle<'_, rig_core::effect::family::Retrieve> {
    /// Scored documents, deserialized on this side of the bus.
    pub fn top_n<T: serde::de::DeserializeOwned>(
        &self,
        req: rig_core::vector_store::request::VectorSearchRequest<
            rig_core::vector_store::request::Filter<serde_json::Value>,
        >,
    ) -> crate::bus::Retrieval<T> {
        self.inner.top_n(req)
    }
}

impl RunHandle<'_, rig_core::effect::family::Completion> {
    /// A unary completion.
    pub fn complete(
        &self,
        request: rig_core::completion::CompletionRequest,
    ) -> crate::bus::Completion {
        self.inner.complete(request)
    }

    /// The model's label as the handler advertises it now.
    pub fn model_ref(&self) -> ModelRef {
        self.inner.model_ref()
    }
}

impl<F: rig_core::effect::Family> std::fmt::Debug for RunHandle<'_, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunHandle")
            .field("key", self.inner.key())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests;
