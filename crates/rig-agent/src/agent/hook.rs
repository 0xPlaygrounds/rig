//! Event-specific hooks for observing and steering an agent run.
//!
//! [`AgentHook`] replaces the old universal event/action pair with one lifecycle
//! method and one action type per event. Unsupported combinations are therefore
//! rejected by the compiler instead of being interpreted at runtime.
//! Hooks are independent of the agent's [`CompletionModel`](crate::completion::CompletionModel):
//! managed response events carry canonical Rig messages, content, usage, and
//! message IDs. Use the direct completion or streaming APIs when a hook-like
//! integration needs the provider's typed raw response.
//!
//! Hooks run in registration order through [`HookStack`]. Model selections
//! chain into later hooks; completion-call [`RequestPatch`] values accumulate
//! and merge; at the dispatch boundary each hook's [`DispatchAction::Patch`]
//! is what the next hook sees, the first [`DispatchAction::Deny`] wins, and
//! each [`OutcomeAction::Replace`] is what the next hook sees. A
//! [`ModelTurnAction::Retry`] or stop action short-circuits the remaining
//! hooks for that event. Nested stacks obey the same rules as flat stacks.
//!
//! **Every effect the run performs crosses one boundary.** The engine
//! dispatches its completions, tool calls, memory loads and appends, and
//! retrievals on the agent's bus, and [`AgentHook::on_dispatch`] /
//! [`AgentHook::on_outcome`] see each of them: a tool call is patched
//! (rewritten arguments), skipped, or stopped *before* it runs and its result
//! replaced or the run stopped *after*; a completion is patched or denied
//! before and observed or replaced after, on either medium. The internal
//! families (`Memory`, `Retrieve`, `Embed`, `Rerank`, `Custom`) are observe-only until
//! a hook opts in through [`AgentHook::observes`]. Register observe-only
//! hooks before steering hooks when every observation is required: a
//! steering stop intentionally prevents later observers from running. A
//! replaced tool result is what the model sees and what result-content
//! telemetry records; execution-outcome metadata describes the result the
//! run keeps.
//!
//! Blocking and streaming agents share model-turn, request, and dispatch
//! resolution. Streaming adds text, reasoning, and tool-call delta
//! observations, but shared lifecycle actions have identical semantics on both
//! surfaces (a streamed completion's outcome is its folded terminal). Streamed
//! deltas are provisional until the model turn is accepted; a retry is
//! surfaced as
//! [`MultiTurnStreamItem::ModelTurnRetried`](crate::agent::MultiTurnStreamItem::ModelTurnRetried)
//! so consumers can discard the rejected turn's deltas.
//!
//! # Example
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
//!
//! # Retrying a completed model turn
//!
//! A hook can reject a tool-free turn and either reuse the same prompt and
//! preceding history with fresh request preparation, or preserve the rejected
//! response and append corrective feedback. Retries use the run's existing
//! total model-call budget. A narrower policy limit belongs to the hook and can
//! be stored in the run-scoped [`Scratchpad`]:
//!
//! ```
//! use std::{collections::HashMap, sync::atomic::{AtomicUsize, Ordering}};
//! use rig_agent::agent::{AgentHook, HookContext, ModelTurnAction, ModelTurnFinished};
//! use rig_core::message::AssistantContent;
//!
//! static NEXT_HOOK_ID: AtomicUsize = AtomicUsize::new(1);
//!
//! #[derive(Clone, Default)]
//! struct RetryCounts(HashMap<usize, usize>);
//!
//! struct RetryOnMarker {
//!     id: usize,
//!     max_retries: usize,
//! }
//!
//! impl RetryOnMarker {
//!     fn new(max_retries: usize) -> Self {
//!         Self {
//!             id: NEXT_HOOK_ID.fetch_add(1, Ordering::Relaxed),
//!             max_retries,
//!         }
//!     }
//! }
//!
//! impl AgentHook for RetryOnMarker {
//!     async fn on_model_turn_finished(
//!         &self,
//!         ctx: &HookContext,
//!         event: ModelTurnFinished<'_>,
//!     ) -> ModelTurnAction {
//!         let rejected = event.content.iter().any(|content| {
//!             matches!(content, AssistantContent::Text(text) if text.text.contains("RETRY"))
//!         });
//!         if !rejected {
//!             return ModelTurnAction::continue_run();
//!         }
//!
//!         let attempt = ctx.scratchpad().update::<RetryCounts, _>(|counts| {
//!             let attempt = counts.0.entry(self.id).or_default();
//!             *attempt += 1;
//!             *attempt
//!         });
//!         if attempt <= self.max_retries {
//!             ModelTurnAction::retry_with_feedback("Return a complete answer.")
//!         } else {
//!             ModelTurnAction::stop("response retry limit exceeded")
//!         }
//!     }
//! }
//! # let _hook = RetryOnMarker::new(2);
//! ```
//!
//! # Retrying a turn the provider cut short
//!
//! [`ModelTurnFinished::finish_reason`] and [`ModelTurnFinished::max_tokens`]
//! carry a turn's termination metadata in portable form, so the common
//! "truncated at the cap, so raise it and go again" policy needs no provider
//! types. `finish_reason` is a normalized [`FinishReason`] — anything outside
//! the shared vocabulary arrives as `Other` in the provider's own spelling
//! rather than as a natural stop, and `None` means the provider reported no
//! reason at all. `max_tokens` is the cap *this* attempt ran under, after the
//! agent's configuration, the runner override, and any merged [`RequestPatch`],
//! so the pair below reads its own escalation back on the retried turn:
//!
//! ```
//! use std::sync::atomic::{AtomicU64, Ordering};
//! use rig_agent::agent::{
//!     AgentHook, CompletionCallAction, CompletionCallEvent, HookContext,
//!     ModelTurnAction, ModelTurnFinished, RequestPatch,
//! };
//! use rig_core::completion::FinishReason;
//! use rig_core::message::AssistantContent;
//!
//! /// Doubles the output cap each time a turn is truncated, up to a ceiling.
//! struct GrowCapOnTruncation {
//!     cap: AtomicU64,
//!     ceiling: u64,
//! }
//!
//! impl AgentHook for GrowCapOnTruncation {
//!     /// Every attempt is prepared afresh, so the current cap is applied here
//!     /// and reported back on that attempt's `ModelTurnFinished`.
//!     async fn on_completion_call(
//!         &self,
//!         _ctx: &HookContext,
//!         _event: CompletionCallEvent<'_>,
//!     ) -> CompletionCallAction {
//!         CompletionCallAction::patch(
//!             RequestPatch::new().max_tokens(self.cap.load(Ordering::Relaxed)),
//!         )
//!     }
//!
//!     async fn on_model_turn_finished(
//!         &self,
//!         _ctx: &HookContext,
//!         event: ModelTurnFinished<'_>,
//!     ) -> ModelTurnAction {
//!         // `truncated_output` covers every reason that means "cut short",
//!         // so a provider reporting a filter stop retries here too.
//!         let truncated = event
//!             .finish_reason
//!             .is_some_and(FinishReason::truncated_output);
//!         // Retrying a turn that carries tool calls is rejected, so a policy
//!         // that might see one has to check before asking.
//!         let has_tool_call = event
//!             .content
//!             .iter()
//!             .any(|content| matches!(content, AssistantContent::ToolCall(_)));
//!         // `max_tokens` is this attempt's own cap: growing past the ceiling
//!         // would be retrying a limit we already know we cannot raise.
//!         let room = event.max_tokens.is_none_or(|cap| cap < self.ceiling);
//!
//!         if truncated && !has_tool_call && room {
//!             let grown = event.max_tokens.map_or(self.ceiling, |cap| {
//!                 cap.saturating_mul(2).min(self.ceiling)
//!             });
//!             self.cap.store(grown, Ordering::Relaxed);
//!             return ModelTurnAction::repeat();
//!         }
//!         ModelTurnAction::continue_run()
//!     }
//! }
//! # let _hook = GrowCapOnTruncation { cap: AtomicU64::new(256), ceiling: 4096 };
//! ```
//!
//! `cargo run -p rig-agent --example retry_on_truncation` runs this policy
//! against a credential-free scripted model whose output genuinely depends on
//! the cap, on both surfaces.

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

/// Run-scoped typed storage shared by hooks — the in-process cross-hook
/// channel.
///
/// # Where hook state belongs
///
/// - **A hook's own private state** belongs in the hook's own fields
///   (`Arc<AtomicUsize>`, `Arc<Mutex<…>>` — the pattern this crate's test
///   probes use). Key by [`HookContext::run_id`] when one instance serves
///   many runs.
/// - **Transient cross-hook state** — one hook writing, another reading,
///   within one run — goes here. Typed entries live only as long as the
///   run's process; a `TypeMap` has no serialization story on purpose.
/// - **Anything that must survive serialization** — an out-of-process
///   approval, a run resumed later, possibly elsewhere — or that must rewind
///   correctly when a host clones/forks a run, goes through
///   [`HookContext::append_entry`]: state that rides the run's record
///   travels, rewinds, and forks with the record; out-of-band state does not.
///
/// [`AgentRun`]: crate::agent::AgentRun
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

/// Run-scoped context supplied to hooks.
#[derive(Debug)]
pub struct HookContext {
    /// The bus the run dispatches through, when it has one.
    dispatcher: Option<rig_bus::Dispatcher>,
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
    /// Every [`RunEntry`] visible to this run — the entries the run carried
    /// in (seeded by the driver at run start) followed by this run's appends,
    /// in append order.
    entries: std::sync::Mutex<Vec<RunEntry>>,
    /// Appends not yet flushed into the [`AgentRun`](crate::agent::AgentRun)
    /// by the driver.
    pending_entries: std::sync::Mutex<Vec<RunEntry>>,
}

impl HookContext {
    pub(crate) fn new(
        is_streaming: bool,
        agent_name: Option<String>,
        dispatcher: Option<rig_bus::Dispatcher>,
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
    fn bus(&self) -> Result<&rig_bus::Dispatcher, ErrorReport> {
        self.dispatcher.as_ref().ok_or_else(|| {
            ErrorReport::new(
                ErrorKind::BusClosed,
                "this hook context was built outside a run and has no bus",
            )
        })
    }

    /// Bind a typed view to a key that carries its family (what the agent
    /// and its registries mint), on the run's bus. The view is scoped to
    /// this context by its lifetime: it routes through the run's driver and
    /// cannot outlive the run — storing it in a field or moving it into a
    /// spawned task does not compile. A dispatch a hook makes this way is
    /// served and recorded but does not re-enter the hook stack. Fails when
    /// the context was built outside a run.
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

    /// Append a durable entry to the run's record.
    ///
    /// The entry is stamped with the current [`turn`](Self::turn) and lands
    /// in the serializable [`AgentRun`](crate::agent::AgentRun) at the next
    /// step boundary — durability holds by construction, with no snapshot or
    /// export moment. Serialization failures surface immediately; nothing is
    /// silently dropped. Fire-and-forget beyond that: no id or handle comes
    /// back.
    ///
    /// The intended default pattern is **snapshot + last-wins**: append a
    /// full state snapshot whenever your state changes, and reconstruct by
    /// reading the most recent one with [`last_entry`](Self::last_entry).
    /// Delta entries folded over [`entries`](Self::entries) fit genuinely
    /// event-shaped state (an approval ledger); the snapshot pattern needs no
    /// fold logic and tolerates replay trivially.
    ///
    /// An entry appended inside `on_run_settled` is not persisted — the run
    /// is already finished; it remains visible to same-process reads only.
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

    /// All entries of `kind` visible to this run: the entries a resumed run
    /// carried in, followed by this run's appends, in append order.
    ///
    /// This *is* the replay: a hook reconstructs state by folding over this
    /// list — and because every read traverses the full list, re-reading is
    /// idempotent by construction.
    pub fn entries(&self, kind: &str) -> Vec<RunEntry> {
        self.entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .filter(|entry| entry.kind == kind)
            .cloned()
            .collect()
    }

    /// The most recent entry of `kind`, if any — the read for the
    /// snapshot-and-read-the-last pattern.
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

pub use crate::run::policy::{InvalidToolCallAction, InvalidToolCallContext, RetryRequest};

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
pub struct CompletionCall<'a> {
    /// Prompt for this turn.
    pub prompt: &'a Message,
    /// History preceding the prompt.
    pub history: &'a [Message],
    /// One-based model-call index.
    pub turn: usize,
}

/// Model-selection event resolved after completion-call hooks and before
/// request preparation.
///
/// The runner default is the first candidate. A [`HookStack`] threads every
/// [`ModelSelectionAction::Select`] into later hooks in registration order, so
/// `selected_model` always reflects all earlier decisions for this event.
///
/// Ordering per `CallModel` step: completion-call hooks resolve first; only if
/// they proceed does this event fire, carrying the merged [`RequestPatch`] in
/// [`request_patch`](Self::request_patch); only after selection resolves does
/// request preparation run against the selected model's captured
/// [`ProviderCapabilities`](crate::completion::ProviderCapabilities), and only
/// then is the attempt issued. Selection therefore runs once per `CallModel`
/// step whose completion-call hooks proceed — including model-turn retries and
/// post-tool calls — and never after a completion-call stop.
///
/// Selection is synchronous, local, and non-blocking: a hook may read and
/// write the run [`Scratchpad`], but must not perform blocking I/O. In-flight
/// attempts never rebind — the selected handle is cloned into the prepared
/// attempt and executes it to completion.
///
/// `previous_model` reflects **issued attempts** only: it advances immediately
/// before the selected model's unary or streaming operation is invoked, so a
/// provider attempt that returns an error still counts, while a
/// completion-call stop, a selection stop, or a request-preparation failure
/// does not. An extraction or run default set via `using_model(...)` is the
/// default candidate for every retry, not a hard pin: selection hooks may
/// override it on each retry.
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
    /// Construct a `ModelSelection` event from its parts.
    ///
    /// Provided so that custom model-selection routers can be unit-tested
    /// outside this crate without restating every field.
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
    /// Why the provider stopped generating this attempt, normalized.
    ///
    /// [`FinishReason`] is the portable vocabulary — `Stop`, `Length`,
    /// `ToolCalls`, `ContentFilter`, and `Other(String)` carrying a provider's
    /// own spelling verbatim for anything outside it — so a hook can decide
    /// whether to accept a turn without naming a provider or touching a raw
    /// response type. [`FinishReason::truncated_output`] is the predicate for
    /// "the provider cut this turn short", which is the usual retry trigger.
    ///
    /// `None` means the provider reported no reason at all, which is a real
    /// outcome for several OpenAI-compatible gateways; it is deliberately not
    /// smoothed into `Stop`, because "finished normally" and "did not say" are
    /// different facts to steer on.
    ///
    /// The value is the one recorded for this attempt's completion call, after
    /// the `Stop`→`ToolCalls` reconciliation that both surfaces apply, so a
    /// provider that reports a bare `stop` on a turn carrying tool calls still
    /// reads as `ToolCalls` here. On a retry this is the retried attempt's own
    /// reason, never a previous attempt's.
    pub finish_reason: Option<&'a FinishReason>,
    /// The output-token cap this exact attempt was prepared with.
    ///
    /// Resolved after the agent's configured value, the runner/request
    /// override, and the merged completion-call
    /// [`RequestPatch`] — so a stateful
    /// completion-call hook that raises the cap for a retry sees its own new
    /// value here on the following turn, not the agent's baseline. `None` means
    /// no cap was sent, so the provider's own default applied.
    ///
    /// Paired with [`finish_reason`](Self::finish_reason) this is what makes a
    /// portable retry-on-truncation decision possible: a hook can tell a turn
    /// cut short at a cap it chose from one cut short at a cap it did not.
    pub max_tokens: Option<u64>,
    /// The provider's own response for this attempt — see
    /// `CompletionResponse::raw` in `rig-core` for the exact meaning of the
    /// payload: the value the model's inherent `raw_completion` /
    /// `raw_stream` would have returned, serialized. Every provider seam
    /// populates it; `Value::Null` only when the response was built without
    /// a provider behind it (a hand-constructed model, a record persisted
    /// before the field). On a retry this is the retried attempt's own,
    /// never a previous attempt's.
    ///
    /// Carried here, and not only on the surface-specific events, for the
    /// same reason identity is: this is the medium-neutral event, so a hook
    /// observing it alone sees the payload for every accepted call on both
    /// surfaces.
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
    /// The stream block this fragment extends — stable across this call's
    /// fragments and equal to the `block_id` of its completed tool call
    /// (the [`DispatchEvent`] for the call).
    /// Provider-issued ids arrive on the completed call.
    pub block_id: &'a BlockId,
    /// Tool name on the first delta.
    pub tool_name: Option<&'a str>,
    /// Newly received argument fragment.
    pub delta: &'a str,
}

/// Pre-run event: the run's initial prompt, before any model call.
///
/// Fired exactly once per run, before the first completion-call hook. The
/// composition rules mirror an input-transform chain: in a [`HookStack`],
/// rewrites chain in registration order — each hook sees the prompt as
/// rewritten by earlier hooks — and the first [`RunStartAction::Stop`] wins,
/// short-circuiting the remaining hooks and terminating the run before any
/// provider call.
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

/// Terminal run event: the run has settled and nothing follows automatically.
///
/// Fired exactly once per run, after the outcome is decided — no retry,
/// further turn, or tool execution will run. This is deliberately distinct
/// from per-turn finishes ([`OutcomeEvent`], [`ModelTurnFinished`]):
/// those can be followed by hook-driven retries or tool turns, while
/// `RunSettled` cannot. On the streaming surface the success case coincides
/// with the run's `FinalResponse` stream item; `RunSettled` additionally
/// covers error termination, which the stream reports as its `Err` item.
#[derive(Clone, Copy)]
pub struct RunSettled<'a> {
    /// How the run ended.
    pub outcome: SettledOutcome<'a>,
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
    RunStart,
    RunSettled,
    CompletionCall,
    ModelTurnFinished,
    InvalidToolCall,
    TextDelta,
    ReasoningDelta,
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
}

/// What a hook decides at the dispatch boundary. Closed on purpose.
#[derive(Debug, Clone)]
#[allow(
    clippy::large_enum_variant,
    reason = "a patch carries a whole effect by design; the common `Proceed` is returned by value once per dispatch"
)]
pub enum DispatchAction {
    /// Dispatch as is.
    Proceed,
    /// Dispatch this effect instead. A patch must keep the family; the
    /// engine rejects a family change as an internal error.
    Patch(EffectKind),
    /// Do not dispatch: the effect resolves failed with this report and
    /// never reaches a handler. For a tool call a report of kind
    /// `Cancelled` cancels the run; any other kind becomes the skipped
    /// result the model sees. For a completion any report fails the turn.
    Deny(ErrorReport),
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
            EffectKind::ToolCall { name, context, .. } => Self::Patch(EffectKind::ToolCall {
                name: name.clone(),
                args: json_utils::serialize_json_value(&args.into()),
                context: context.clone(),
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
        match self.kind {
            EffectKind::ToolCall { context, .. } => Some(context),
            _ => None,
        }
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
}

/// What a hook decides after an effect resolved. Closed on purpose.
#[derive(Debug, Clone)]
#[allow(
    clippy::large_enum_variant,
    reason = "a replacement carries a whole outcome by design; the common `Proceed` is returned by value once per dispatch"
)]
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
    pub fn stop(reason: impl Into<String>) -> Self {
        Self::Replace(Err(ErrorReport::new(ErrorKind::Cancelled, reason)))
    }

    /// Replace the model-visible output of a tool result, keeping the
    /// result's status and the dispatch context. `Proceed` when `event`
    /// did not resolve to a tool result.
    pub fn rewrite_tool_output(event: &OutcomeEvent<'_>, output: ToolOutput) -> Self {
        match event.outcome {
            Ok(Outcome::ToolResult { result, context }) => Self::Replace(Ok(Outcome::ToolResult {
                result: result.clone().with_output(output),
                context: context.clone(),
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
        match self.outcome {
            Ok(Outcome::ToolResult { context, .. }) => Some(context),
            _ => None,
        }
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
    /// Runs once before the run's first model call, seeing the initial prompt.
    ///
    /// The hook may rewrite the prompt or stop the run before any provider
    /// call. In a [`HookStack`], rewrites chain in registration order and the
    /// first stop wins — see [`RunStart`]. The default action starts the run
    /// with the current prompt.
    fn on_run_start(
        &self,
        _ctx: &HookContext,
        _event: RunStart<'_>,
    ) -> impl Future<Output = RunStartAction> + WasmCompatSend {
        async { RunStartAction::Continue }
    }

    /// Runs once after the run settles: its outcome — final response or
    /// terminal error — is decided, and no retry, further turn, or tool
    /// execution will follow. Observe-only; the run is already over.
    ///
    /// [`HookContext::entries`] sees every entry appended during the run
    /// (the driver flushes before settling), but an entry appended *inside*
    /// this hook is not persisted — the run is finished.
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
    /// `CallModel` step whose completion-call hooks proceed — including
    /// retries and post-tool calls — never after a completion-call stop, and
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
        _event: CompletionCall<'_>,
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

    /// Observation interest hint, primarily for high-frequency deltas. The
    /// internal `Memory`/`Retrieve`/`Embed`/`Rerank`/`Custom` dispatch events are
    /// off by default: no hook saw those calls before the bus, so a hook
    /// that wants to gate them opts in here.
    ///
    /// This is a gate, not a hint, for the dispatch-boundary events: a
    /// `false` for `CompletionDispatch` or `ToolDispatch` silences
    /// [`AgentHook::on_dispatch`] and [`AgentHook::on_outcome`] for that
    /// family — the hook can no longer deny a tool call. An override that
    /// only wants to drop deltas answers `false` for the delta kinds alone
    /// and leaves every `*Dispatch` kind at the default.
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
            CompletionCall,
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
    /// The type name of every hook, in registration order, nested stacks
    /// flattened: what an effect log's header records as the program.
    names: Vec<&'static str>,
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

    /// Appends a hook to the end of the stack's registration order.
    ///
    /// The stack names the hook by its type's last path segment (`Tagger`,
    /// not `my_crate::hooks::Tagger`): the name is the program identity an
    /// effect log records, and the same program compiled into another
    /// crate — a test suite replaying a golden its producer recorded — must
    /// name the same hooks.
    pub fn push<H: AgentHook + 'static>(&mut self, hook: H) {
        // A nested stack contributes its members' names, flattened in
        // order, so two stacks that run the same hooks name the same program.
        match (&hook as &dyn std::any::Any).downcast_ref::<HookStack>() {
            Some(nested) => self.names.extend(nested.names.iter().copied()),
            None => self.names.push(short_type_name::<H>()),
        }
        self.hooks.push(Arc::new(hook));
    }

    /// The type names of the hooks in registration order (nested stacks
    /// flattened) — the program identity an effect log records.
    pub fn names(&self) -> Vec<String> {
        self.names.iter().map(|name| (*name).to_owned()).collect()
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
        // Every hook observes the terminal event; there is nothing to
        // short-circuit on since the run is already over.
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
        event: CompletionCall<'_>,
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
                DispatchAction::Patch(next) => patched = Some(next),
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
                OutcomeAction::Replace(next) => replaced = Some(next),
            }
        }
        replaced.map_or(OutcomeAction::Proceed, OutcomeAction::Replace)
    }

    fn observes(&self, kind: StepEventKind) -> bool {
        self.hooks.iter().any(|hook| hook.observes(kind))
    }
}

/// A typed view a hook bound through its [`HookContext`], scoped to the
/// run by its lifetime: it is `!'static`, so a hook cannot store it in a
/// field or move it into a spawned task — the compiler refuses. It offers
/// the family-generic API of [`Handle`](rig_bus::Handle) by
/// delegation rather than `Deref` (a `Deref` to the `Clone` handle would
/// hand back an owned `'static` view through `.clone()`, which is the one
/// thing this type exists to prevent). A host that wants an owned handle
/// takes it from a [`Dispatcher`](rig_bus::Dispatcher) it holds
/// itself.
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
    inner: rig_bus::Handle<F>,
    _run: std::marker::PhantomData<&'ctx HookContext>,
}

impl<'ctx, F: rig_core::effect::Family> RunHandle<'ctx, F> {
    fn scoped(inner: rig_bus::Handle<F>) -> Self {
        Self {
            inner,
            _run: std::marker::PhantomData,
        }
    }

    /// Dispatch a typed request of this family.
    pub fn dispatch(&self, request: F::Request) -> rig_bus::Typed<F> {
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
    ) -> rig_bus::Retrieval<T> {
        self.inner.top_n(req)
    }
}

impl RunHandle<'_, rig_core::effect::family::Completion> {
    /// A unary completion.
    pub fn complete(
        &self,
        request: rig_core::completion::CompletionRequest,
    ) -> rig_bus::Completion {
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
