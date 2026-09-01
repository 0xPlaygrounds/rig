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
//! Hooks run in registration order through [`HookStack`]. Model selections,
//! tool-call argument rewrites, and tool-result presentation rewrites chain into
//! later hooks; completion-call [`RequestPatch`] values accumulate and merge.
//! A [`ModelTurnAction::Retry`] or stop action short-circuits the remaining
//! hooks for that event. Nested stacks obey the same rules as flat stacks,
//! including preserving an argument rewrite when an inner stack later skips or
//! stops.
//!
//! Register observe-only hooks before steering hooks when every observation is
//! required: a steering stop intentionally prevents later observers from
//! running. Tool-result rewrites change the effective `presentation` sent to
//! the model and recorded as result-content telemetry. The
//! [`ToolResultEvent::raw_result`] and its [`ToolResultEvent::tool_context`]
//! remain unchanged for policy decisions and execution-outcome metadata. A
//! tool-result stop omits result content from telemetry.
//!
//! Blocking and streaming agents share model-turn, request, tool-call, and
//! tool-result resolution. Streaming adds text, reasoning, and tool-call delta
//! observations, but shared lifecycle actions have identical semantics on both
//! surfaces. Streamed deltas are provisional until the model turn is accepted;
//! a retry is surfaced as
//! [`MultiTurnStreamItem::ModelTurnRetried`](crate::agent::MultiTurnStreamItem::ModelTurnRetried)
//! so consumers can discard the rejected turn's deltas.
//!
//! # Example
//!
//! ```
//! use rig_agent::agent::{
//!     AgentHook, CompletionResponseEvent, HookContext, ObservationAction,
//! };
//!
//! struct ResponseLogger;
//!
//! impl AgentHook for ResponseLogger {
//!     async fn on_completion_response(
//!         &self,
//!         _ctx: &HookContext,
//!         event: CompletionResponseEvent<'_>,
//!     ) -> ObservationAction {
//!         println!(
//!             "message {:?}: {:?} ({:?})",
//!             event.message_id, event.content, event.usage
//!         );
//!         ObservationAction::continue_run()
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

use rig_core::id::InternalCallId;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{future::Future, sync::Arc};

use rig_core::tool::context::TypeMap;
use rig_core::{
    completion::FinishReason,
    message::{AssistantContent, Message},
    wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync},
};

use crate::{
    agent::ModelHandle,
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

type ToolCallRewriteFrameMap = HashMap<InternalCallId, Vec<Option<serde_json::Value>>>;

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
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn begin(&self, internal_call_id: InternalCallId) -> ToolCallResolutionFrame<'_> {
        self.lock().entry(internal_call_id).or_default().push(None);
        ToolCallResolutionFrame {
            frames: self,
            internal_call_id,
            active: true,
        }
    }

    fn record(&self, internal_call_id: InternalCallId, rewrite: serde_json::Value) {
        if let Some(frame) = self
            .lock()
            .get_mut(&internal_call_id)
            .and_then(|frames| frames.last_mut())
        {
            *frame = Some(rewrite);
        }
    }

    fn finish(&self, internal_call_id: InternalCallId) -> Option<serde_json::Value> {
        let mut frames = self.lock();
        let (rewrite, remove_entry) =
            frames
                .get_mut(&internal_call_id)
                .map_or((None, false), |frames| {
                    let rewrite = frames.pop().flatten();
                    (rewrite, frames.is_empty())
                });
        if remove_entry {
            frames.remove(&internal_call_id);
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
    internal_call_id: InternalCallId,
    active: bool,
}

impl ToolCallResolutionFrame<'_> {
    fn finish(mut self) -> Option<serde_json::Value> {
        self.active = false;
        self.frames.finish(self.internal_call_id)
    }
}

impl Drop for ToolCallResolutionFrame<'_> {
    fn drop(&mut self) {
        if self.active {
            self.frames.finish(self.internal_call_id);
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
    /// Every [`RunEntry`] visible to this run — the entries the run carried
    /// in (seeded by the driver at run start) followed by this run's appends,
    /// in append order.
    entries: std::sync::Mutex<Vec<RunEntry>>,
    /// Appends not yet flushed into the [`AgentRun`](crate::agent::AgentRun)
    /// by the driver.
    pending_entries: std::sync::Mutex<Vec<RunEntry>>,
}

impl HookContext {
    pub(crate) fn new(is_streaming: bool, agent_name: Option<String>) -> Self {
        Self {
            run_id: RunId::new(),
            turn: AtomicUsize::new(0),
            is_streaming,
            agent_name,
            scratchpad: Scratchpad::default(),
            tool_call_rewrite_frames: ToolCallRewriteFrames::default(),
            entries: std::sync::Mutex::new(Vec::new()),
            pending_entries: std::sync::Mutex::new(Vec::new()),
        }
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

    fn begin_tool_call_resolution(
        &self,
        internal_call_id: InternalCallId,
    ) -> ToolCallResolutionFrame<'_> {
        self.tool_call_rewrite_frames.begin(internal_call_id)
    }

    fn record_tool_call_rewrite(
        &self,
        internal_call_id: InternalCallId,
        rewrite: serde_json::Value,
    ) {
        self.tool_call_rewrite_frames
            .record(internal_call_id, rewrite);
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
    pub previous_model: Option<&'a ModelHandle>,
    /// Runner default used as the initial candidate for this call.
    pub default_model: &'a ModelHandle,
    /// Candidate after all earlier model-selection hooks.
    pub selected_model: &'a ModelHandle,
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
        previous_model: Option<&'a ModelHandle>,
        default_model: &'a ModelHandle,
        selected_model: &'a ModelHandle,
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

/// Canonical non-streaming completion response event.
#[derive(Clone, Copy)]
pub struct CompletionResponse<'a> {
    /// Prompt sent for this turn.
    pub prompt: &'a Message,
    /// Canonical assistant content returned for this turn.
    pub content: &'a Vec<AssistantContent>,
    /// Usage reported for this turn.
    pub usage: Usage,
    /// Provider-assigned message ID, when available. Always equal to
    /// [`identity`](Self::identity)`.message_id`; kept as a field for
    /// continuity with pre-identity hooks.
    pub message_id: Option<&'a str>,
    /// This exact attempt's response identity metadata (message-scoped,
    /// response-scoped, and transport request ids).
    pub identity: &'a ResponseIdentity,
    /// The provider's own response for this attempt — see
    /// `CompletionResponse::raw` in `rig-core` for the exact meaning of the
    /// payload: the value the model's inherent `raw_completion` /
    /// `raw_stream` would have returned, serialized. Every provider seam
    /// populates it; `Value::Null` only when the response was built without
    /// a provider behind it (a hand-constructed model, a record persisted
    /// before the field). On a retry this is the retried attempt's own,
    /// never a previous attempt's.
    pub raw: &'a serde_json::Value,
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
    /// This exact attempt's response identity metadata. Fired for every
    /// completed model call on both surfaces — including streamed tool-only
    /// and reasoning-only turns, which fire no [`StreamResponseFinish`] — so
    /// a provider-neutral hook observing this event alone records identity
    /// for every accepted call. On a retry, this is the retried attempt's own
    /// identity, never a previous attempt's.
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

/// Pre-execution tool event.
#[derive(Clone, Copy)]
pub struct ToolCall<'a> {
    /// Tool name.
    pub tool_name: &'a str,
    /// Durable tool-call id: the provider's when it issued one, else rig's
    /// minted handle.
    pub tool_call_id: Option<&'a str>,
    /// Rig correlation id.
    pub internal_call_id: InternalCallId,
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
    /// Durable tool-call id: the provider's when it issued one, else rig's
    /// minted handle.
    pub tool_call_id: Option<&'a str>,
    /// Rig correlation id.
    pub internal_call_id: InternalCallId,
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

/// Streaming reasoning delta.
#[derive(Clone, Copy)]
pub struct ReasoningDelta<'a> {
    /// Rig-generated correlator for this reasoning part. It is stable across
    /// the part's deltas and eventual completed reasoning item, but is never
    /// persisted as a provider-issued reasoning id.
    pub id: &'a str,
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
    /// Rig correlation id — stable across this call's fragments and its
    /// completed [`ToolCall`], unique per run. Provider-issued ids arrive on
    /// the completed call; no provider id (and no stream-internal key) is
    /// available or rendered at delta time.
    pub internal_call_id: InternalCallId,
    /// Tool name on the first delta.
    pub tool_name: Option<&'a str>,
    /// Newly received argument fragment.
    pub delta: &'a str,
}

/// Canonical streaming response-finish event.
#[derive(Clone, Copy)]
pub struct StreamResponseFinish<'a> {
    /// Prompt sent for this turn.
    pub prompt: &'a Message,
    /// Canonical assistant content aggregated for this turn.
    pub content: &'a Vec<AssistantContent>,
    /// Usage reported for this turn.
    pub usage: Usage,
    /// Provider-assigned message ID, when available. Always equal to
    /// [`identity`](Self::identity)`.message_id`; kept as a field for
    /// continuity with pre-identity hooks.
    pub message_id: Option<&'a str>,
    /// This exact attempt's response identity metadata (message-scoped,
    /// response-scoped, and transport request ids).
    pub identity: &'a ResponseIdentity,
    /// The provider's own response for this attempt — see
    /// `CompletionResponse::raw` in `rig-core` for the exact meaning of the
    /// payload: the value the model's inherent `raw_completion` /
    /// `raw_stream` would have returned, serialized. Every provider seam
    /// populates it; `Value::Null` only when the response was built without
    /// a provider behind it (a hand-constructed model, a record persisted
    /// before the field). On a retry this is the retried attempt's own,
    /// never a previous attempt's.
    pub raw: &'a serde_json::Value,
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
/// from per-turn finishes ([`ModelTurnFinished`], [`StreamResponseFinish`]):
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
    CompletionResponse,
    ModelTurnFinished,
    InvalidToolCall,
    ToolCall,
    ToolResult,
    TextDelta,
    ReasoningDelta,
    ToolCallDelta,
    StreamResponseFinish,
}

pub use rig_core::completion::patch::RequestPatch;

/// Action for model-selection hooks.
#[derive(Debug, Clone)]
pub enum ModelSelectionAction {
    /// Keep the candidate supplied to this hook.
    Continue,
    /// Replace the candidate and pass it to later hooks.
    Select(ModelHandle),
    /// Stop the run before request preparation or model execution.
    Stop(String),
}

impl ModelSelectionAction {
    /// Keeps the current model candidate.
    pub fn continue_run() -> Self {
        Self::Continue
    }

    /// Selects `model` and passes it to later hooks.
    pub fn select(model: ModelHandle) -> Self {
        Self::Select(model)
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
    /// already-constructed [`ModelHandle`] values and may read or write the
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

    /// Observes a completed model response.
    ///
    /// The default action continues the run.
    fn on_completion_response(
        &self,
        _ctx: &HookContext,
        _event: CompletionResponse<'_>,
    ) -> impl Future<Output = ObservationAction> + WasmCompatSend {
        async { ObservationAction::Continue }
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

    /// Observes a completed streaming response in canonical Rig form.
    ///
    /// The default action continues the run.
    fn on_stream_response_finish(
        &self,
        _ctx: &HookContext,
        _event: StreamResponseFinish<'_>,
    ) -> impl Future<Output = ObservationAction> + WasmCompatSend {
        async { ObservationAction::Continue }
    }

    /// Observation interest hint, primarily for high-frequency deltas.
    fn observes(&self, _kind: StepEventKind) -> bool {
        true
    }
}

impl AgentHook for () {
    fn observes(&self, _kind: StepEventKind) -> bool {
        false
    }
}

/// The erased hook events whose dispatch is a plain `Box::pin(self.on_*(..))`.
/// `model_select` (sync), `invalid_tool_call` (borrowed event), and `tool_call`
/// (wraps the rewrite-salvage frame) are hand-written below.
macro_rules! for_each_boxed_hook_event {
    ($m:ident) => {
        $m!(
            completion_call,
            on_completion_call,
            CompletionCall,
            CompletionCallAction
        );
        $m!(
            completion_response,
            on_completion_response,
            CompletionResponse,
            ObservationAction
        );
        $m!(
            model_turn_finished,
            on_model_turn_finished,
            ModelTurnFinished,
            ModelTurnAction
        );
        $m!(
            tool_result,
            on_tool_result,
            ToolResultEvent,
            ToolResultAction
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
        $m!(
            stream_response_finish,
            on_stream_response_finish,
            StreamResponseFinish,
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
    fn tool_call<'a>(
        &'a self,
        ctx: &'a HookContext,
        event: ToolCall<'a>,
    ) -> WasmBoxedFuture<'a, (ToolCallAction, Option<serde_json::Value>)>;
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
    fn tool_call<'a>(
        &'a self,
        ctx: &'a HookContext,
        event: ToolCall<'a>,
    ) -> WasmBoxedFuture<'a, (ToolCallAction, Option<serde_json::Value>)> {
        Box::pin(async move {
            // Only `on_tool_call` is public dispatch. A nested `HookStack`
            // records terminal-path rewrite state into this private frame.
            let frame = ctx.begin_tool_call_resolution(event.internal_call_id);
            let action = self.on_tool_call(ctx, event).await;
            (action, frame.finish())
        })
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
}

impl std::fmt::Debug for HookStack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HookStack")
            .field("len", &self.hooks.len())
            .finish()
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
    pub fn push<H: AgentHook + 'static>(&mut self, hook: H) {
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
/// `on_invalid_tool_call`, `on_tool_call`, `on_tool_result`) stay hand-written.
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
        on_completion_response, completion_response, CompletionResponse, ObservationAction;
        on_model_turn_finished, model_turn_finished, ModelTurnFinished, ModelTurnAction;
        on_text_delta, text_delta, TextDelta, ObservationAction;
        on_reasoning_delta, reasoning_delta, ReasoningDelta, ObservationAction;
        on_tool_call_delta, tool_call_delta, ToolCallDelta, ObservationAction;
        on_stream_response_finish, stream_response_finish, StreamResponseFinish, ObservationAction;
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
    fn observes(&self, kind: StepEventKind) -> bool {
        self.hooks.iter().any(|hook| hook.observes(kind))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::{ToolErrorKind, ToolExecutionError};

    /// Rewrites the run-start prompt by appending its tag; used to observe
    /// rewrite chaining across a stack.
    struct StartRewriter(&'static str);
    impl AgentHook for StartRewriter {
        async fn on_run_start(&self, _ctx: &HookContext, event: RunStart<'_>) -> RunStartAction {
            let current = match event.prompt {
                Message::User { .. } => event.prompt.rag_text().expect("test prompts carry text"),
                _ => panic!("run-start prompts are user messages"),
            };
            RunStartAction::rewrite(Message::user(format!("{current}{}", self.0)))
        }
    }

    struct StartStopper;
    impl AgentHook for StartStopper {
        async fn on_run_start(&self, _ctx: &HookContext, _event: RunStart<'_>) -> RunStartAction {
            RunStartAction::stop("blocked at start")
        }
    }

    #[tokio::test]
    async fn run_start_rewrites_chain_in_registration_order() {
        let mut stack = HookStack::with(StartRewriter("-a"));
        stack.push(StartRewriter("-b"));
        let ctx = HookContext::new(false, None);
        let prompt = Message::user("p");
        let action = stack
            .on_run_start(
                &ctx,
                RunStart {
                    prompt: &prompt,
                    history: &[],
                },
            )
            .await;
        match action {
            RunStartAction::Rewrite(message) => {
                // The second hook saw the first hook's rewrite.
                assert_eq!(message.rag_text().expect("text"), "p-a-b");
            }
            other => panic!("expected a chained rewrite, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn run_start_first_stop_wins_and_short_circuits() {
        let mut stack = HookStack::with(StartRewriter("-a"));
        stack.push(StartStopper);
        // This rewriter must never run.
        struct Panicker;
        impl AgentHook for Panicker {
            async fn on_run_start(
                &self,
                _ctx: &HookContext,
                _event: RunStart<'_>,
            ) -> RunStartAction {
                panic!("a stop must short-circuit later hooks");
            }
        }
        stack.push(Panicker);
        let ctx = HookContext::new(false, None);
        let prompt = Message::user("p");
        let action = stack
            .on_run_start(
                &ctx,
                RunStart {
                    prompt: &prompt,
                    history: &[],
                },
            )
            .await;
        assert_eq!(action, RunStartAction::Stop("blocked at start".into()));
    }

    #[test]
    fn append_entry_stamps_the_current_turn_and_reads_replay_in_order() {
        let ctx = HookContext::new(false, None);
        // A resumed run's carried entries come first.
        ctx.seed_entries(&[RunEntry {
            kind: "counter".into(),
            turn: 2,
            value: serde_json::json!(2),
        }]);

        ctx.set_turn(3);
        ctx.append_entry("counter", &3u64).expect("serializable");
        ctx.append_entry("other", &()).expect("null marker");

        let counters = ctx.entries("counter");
        assert_eq!(
            counters.iter().map(|e| e.turn).collect::<Vec<_>>(),
            [2, 3],
            "seeded entries precede this run's appends"
        );
        // Last-wins snapshot read.
        let last = ctx.last_entry("counter").expect("appended");
        assert_eq!(last.turn, 3);
        assert_eq!(last.value, serde_json::json!(3));
        assert!(ctx.last_entry("absent").is_none());

        // Only this run's appends are pending for the driver to flush —
        // seeded entries are already in the run.
        let pending = ctx.drain_pending_entries();
        assert_eq!(
            pending.iter().map(|e| e.kind.as_str()).collect::<Vec<_>>(),
            ["counter", "other"]
        );
        // Draining does not affect reads, and is not repeatable.
        assert_eq!(ctx.entries("counter").len(), 2);
        assert!(ctx.drain_pending_entries().is_empty());
    }

    struct Patcher(f64);
    impl AgentHook for Patcher {
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

    impl AgentHook for CallRewriter {
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
                    internal_call_id: InternalCallId::new(),
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

    impl AgentHook for ResultRewriter {
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
                    internal_call_id: InternalCallId::new(),
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

    impl AgentHook for StopThenCount {
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
                    internal_call_id: InternalCallId::new(),
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
    use serde_json::{Value, json};

    fn ctx() -> HookContext {
        HookContext::new(false, Some("test-agent".to_string()))
    }

    fn model(label: &str) -> ModelHandle {
        ModelHandle::named(label, crate::test_utils::MockCompletionModel::default())
    }

    enum RouteDecision {
        Continue,
        Select(ModelHandle),
        Stop,
    }

    type RouteLog = Arc<Mutex<Vec<(&'static str, Option<String>)>>>;

    struct RouteRecorder {
        label: &'static str,
        log: RouteLog,
        decision: RouteDecision,
    }

    impl AgentHook for RouteRecorder {
        fn on_model_select(
            &self,
            _ctx: &HookContext,
            event: ModelSelection<'_>,
        ) -> ModelSelectionAction {
            self.log
                .lock()
                .expect("route log")
                .push((self.label, event.selected_model.label().map(str::to_owned)));
            match &self.decision {
                RouteDecision::Continue => ModelSelectionAction::continue_run(),
                RouteDecision::Select(model) => ModelSelectionAction::select(model.clone()),
                RouteDecision::Stop => ModelSelectionAction::stop("routing stopped"),
            }
        }
    }

    fn model_selection<'a>(
        prompt: &'a Message,
        default_model: &'a ModelHandle,
    ) -> ModelSelection<'a> {
        ModelSelection {
            prompt,
            history: &[],
            request_patch: None,
            previous_model: None,
            default_model,
            selected_model: default_model,
        }
    }

    #[test]
    fn model_selections_chain_in_registration_order_and_last_wins() {
        let default = model("default");
        let first = model("first");
        let last = model("last");
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut stack = HookStack::with(RouteRecorder {
            label: "continue",
            log: log.clone(),
            decision: RouteDecision::Continue,
        });
        stack.push(RouteRecorder {
            label: "first",
            log: log.clone(),
            decision: RouteDecision::Select(first),
        });
        stack.push(RouteRecorder {
            label: "last",
            log: log.clone(),
            decision: RouteDecision::Select(last),
        });
        let prompt = Message::user("route");

        let action = stack.on_model_select(&ctx(), model_selection(&prompt, &default));

        let ModelSelectionAction::Select(selected) = action else {
            panic!("stack should select the last candidate");
        };
        assert_eq!(selected.label(), Some("last"));
        assert_eq!(
            log.lock().expect("route log").as_slice(),
            &[
                ("continue", Some("default".to_owned())),
                ("first", Some("default".to_owned())),
                ("last", Some("first".to_owned())),
            ]
        );
    }

    #[test]
    fn model_selection_stop_short_circuits_later_hooks() {
        let default = model("default");
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut stack = HookStack::with(RouteRecorder {
            label: "stop",
            log: log.clone(),
            decision: RouteDecision::Stop,
        });
        stack.push(RouteRecorder {
            label: "later",
            log: log.clone(),
            decision: RouteDecision::Select(model("later")),
        });
        let prompt = Message::user("route");

        assert!(matches!(
            stack.on_model_select(&ctx(), model_selection(&prompt, &default)),
            ModelSelectionAction::Stop(reason) if reason == "routing stopped"
        ));
        assert_eq!(
            log.lock().expect("route log").as_slice(),
            &[("stop", Some("default".to_owned()))]
        );
    }

    #[test]
    fn nested_model_selection_stacks_preserve_candidate_chaining() {
        let default = model("default");
        let log = Arc::new(Mutex::new(Vec::new()));
        let inner = HookStack::with(RouteRecorder {
            label: "inner",
            log: log.clone(),
            decision: RouteDecision::Select(model("inner")),
        });
        let mut outer = HookStack::with(RouteRecorder {
            label: "outer-before",
            log: log.clone(),
            decision: RouteDecision::Select(model("outer")),
        });
        outer.push(inner);
        outer.push(RouteRecorder {
            label: "outer-after",
            log: log.clone(),
            decision: RouteDecision::Continue,
        });
        let prompt = Message::user("route");

        let action = outer.on_model_select(&ctx(), model_selection(&prompt, &default));

        let ModelSelectionAction::Select(selected) = action else {
            panic!("nested stack should preserve the inner selection");
        };
        assert_eq!(selected.label(), Some("inner"));
        assert_eq!(
            log.lock().expect("route log").as_slice(),
            &[
                ("outer-before", Some("default".to_owned())),
                ("inner", Some("outer".to_owned())),
                ("outer-after", Some("inner".to_owned())),
            ]
        );
    }

    #[test]
    fn nested_model_selection_stack_without_a_selection_preserves_outer_candidate() {
        let default = model("default");
        let log = Arc::new(Mutex::new(Vec::new()));
        let inner = HookStack::with(RouteRecorder {
            label: "inner-continue",
            log: log.clone(),
            decision: RouteDecision::Continue,
        });
        let mut outer = HookStack::with(RouteRecorder {
            label: "outer-select",
            log: log.clone(),
            decision: RouteDecision::Select(model("outer")),
        });
        outer.push(inner);
        outer.push(RouteRecorder {
            label: "outer-after",
            log: log.clone(),
            decision: RouteDecision::Continue,
        });
        let prompt = Message::user("route");

        let action = outer.on_model_select(&ctx(), model_selection(&prompt, &default));

        let ModelSelectionAction::Select(selected) = action else {
            panic!("outer selection should survive a continuing nested stack");
        };
        assert_eq!(selected.label(), Some("outer"));
        assert_eq!(
            log.lock().expect("route log").as_slice(),
            &[
                ("outer-select", Some("default".to_owned())),
                ("inner-continue", Some("outer".to_owned())),
                ("outer-after", Some("outer".to_owned())),
            ]
        );
    }

    #[test]
    fn nested_model_selection_stop_short_circuits_the_outer_stack() {
        let default = model("default");
        let log = Arc::new(Mutex::new(Vec::new()));
        let inner = HookStack::with(RouteRecorder {
            label: "inner-stop",
            log: log.clone(),
            decision: RouteDecision::Stop,
        });
        let mut outer = HookStack::with(RouteRecorder {
            label: "outer-before",
            log: log.clone(),
            decision: RouteDecision::Select(model("outer")),
        });
        outer.push(inner);
        outer.push(RouteRecorder {
            label: "outer-after",
            log: log.clone(),
            decision: RouteDecision::Select(model("unreachable")),
        });
        let prompt = Message::user("route");

        assert!(matches!(
            outer.on_model_select(&ctx(), model_selection(&prompt, &default)),
            ModelSelectionAction::Stop(reason) if reason == "routing stopped"
        ));
        assert_eq!(
            log.lock().expect("route log").as_slice(),
            &[
                ("outer-before", Some("default".to_owned())),
                ("inner-stop", Some("outer".to_owned())),
            ]
        );
    }

    struct ToolRecorder {
        label: u32,
        log: Arc<Mutex<Vec<u32>>>,
        stop: bool,
    }
    impl AgentHook for ToolRecorder {
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
    impl AgentHook for ObservationRecorder {
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

        async fn on_reasoning_delta(
            &self,
            _ctx: &HookContext,
            _event: ReasoningDelta<'_>,
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
    impl AgentHook for ObservesOnly {
        fn observes(&self, kind: StepEventKind) -> bool {
            kind == self.0
        }
    }

    struct InvalidResponder {
        action: InvalidToolCallAction,
        calls: Arc<AtomicUsize>,
    }
    impl AgentHook for InvalidResponder {
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
    impl AgentHook for Patcher {
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
            internal_call_id: InternalCallId::new(),
            args: "{}",
        }
    }
    fn completion_call_event() -> CompletionCall<'static> {
        static PROMPT: std::sync::OnceLock<rig_core::message::Message> = std::sync::OnceLock::new();
        CompletionCall {
            prompt: PROMPT.get_or_init(|| rig_core::message::Message::user("hi")),
            history: &[],
            turn: 1,
        }
    }

    fn invalid_tool_call_context() -> InvalidToolCallContext {
        InvalidToolCallContext {
            tool_name: "unknown".into(),
            tool_call_id: Some("tc1".into()),
            internal_call_id: Some(InternalCallId::new()),
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
        let mut stack = HookStack::with(ToolRecorder {
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
        let mut stack = HookStack::with(ToolRecorder {
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
        let mut stack = HookStack::with(ObservationRecorder {
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
    async fn reasoning_delta_observation_preserves_nested_order_and_stop() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut inner = HookStack::with(ObservationRecorder {
            label: 1,
            log: log.clone(),
            stop: false,
        });
        inner.push(ObservationRecorder {
            label: 2,
            log: log.clone(),
            stop: true,
        });
        let mut outer = HookStack::with(inner);
        outer.push(ObservationRecorder {
            label: 3,
            log: log.clone(),
            stop: false,
        });

        assert!(matches!(
            outer
                .on_reasoning_delta(
                    &ctx(),
                    ReasoningDelta {
                        id: "corr_1",
                        provider_id: Some("rs_1"),
                        delta: "think",
                        aggregated: "think",
                    },
                )
                .await,
            ObservationAction::Stop(_)
        ));
        assert_eq!(*log.lock().expect("log"), vec![1, 2]);
    }

    #[tokio::test]
    async fn explicit_fail_short_circuits_later_invalid_tool_hooks() {
        let fail_calls = Arc::new(AtomicUsize::new(0));
        let retry_calls = Arc::new(AtomicUsize::new(0));
        let mut stack = HookStack::with(InvalidResponder {
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
        let mut stack = HookStack::with(());
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
        let mut stack = HookStack::with(Patcher {
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
        let mut stopped = HookStack::with(Patcher {
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
        let mut inner = HookStack::with(Patcher {
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
        let mut outer = HookStack::with(inner);
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
        let mut stack = HookStack::with(ObservesOnly(StepEventKind::ToolCall));
        stack.push(ObservesOnly(StepEventKind::ToolResult));
        assert!(<HookStack as AgentHook>::observes(
            &stack,
            StepEventKind::ToolCall
        ));
        assert!(<HookStack as AgentHook>::observes(
            &stack,
            StepEventKind::ToolResult
        ));
        assert!(!<HookStack as AgentHook>::observes(
            &stack,
            StepEventKind::TextDelta
        ));
    }

    #[test]
    fn empty_stack_observes_nothing() {
        let empty = HookStack::new();
        assert!(empty.is_empty());
        assert!(!<HookStack as AgentHook>::observes(
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
            StepEventKind::ReasoningDelta,
            StepEventKind::ToolCallDelta,
            StepEventKind::StreamResponseFinish,
        ] {
            assert!(!<() as AgentHook>::observes(&(), kind));
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
        assert!(context.run_id().to_raw() > 0);
    }

    struct RewriteHook(Value);
    impl AgentHook for RewriteHook {
        async fn on_tool_call(&self, _: &HookContext, _: ToolCall<'_>) -> ToolCallAction {
            ToolCallAction::rewrite(self.0.clone())
        }
    }
    struct SkipHook;
    impl AgentHook for SkipHook {
        async fn on_tool_call(&self, _: &HookContext, _: ToolCall<'_>) -> ToolCallAction {
            ToolCallAction::skip("denied")
        }
    }
    struct StopHook;
    impl AgentHook for StopHook {
        async fn on_tool_call(&self, _: &HookContext, _: ToolCall<'_>) -> ToolCallAction {
            ToolCallAction::stop("stop")
        }
    }
    #[derive(Clone, Default)]
    struct ArgsSpy(Arc<Mutex<Vec<String>>>);
    impl AgentHook for ArgsSpy {
        async fn on_tool_call(&self, _: &HookContext, event: ToolCall<'_>) -> ToolCallAction {
            self.0.lock().unwrap().push(event.args.into());
            ToolCallAction::run()
        }
    }

    struct OnToolCallOnly(Arc<AtomicUsize>);
    impl AgentHook for OnToolCallOnly {
        async fn on_tool_call(&self, _: &HookContext, _: ToolCall<'_>) -> ToolCallAction {
            self.0.fetch_add(1, Ordering::Relaxed);
            ToolCallAction::skip("called")
        }
    }

    struct YieldingRewriteFromCallId;
    impl AgentHook for YieldingRewriteFromCallId {
        async fn on_tool_call(&self, _: &HookContext, event: ToolCall<'_>) -> ToolCallAction {
            tokio::task::yield_now().await;
            ToolCallAction::rewrite(json!({"call_id": event.internal_call_id}))
        }
    }

    struct YieldingSkip;
    impl AgentHook for YieldingSkip {
        async fn on_tool_call(&self, _: &HookContext, _: ToolCall<'_>) -> ToolCallAction {
            tokio::task::yield_now().await;
            ToolCallAction::skip("denied")
        }
    }

    async fn resolve(stack: &HookStack) -> (ToolCallAction, Option<Value>) {
        stack.resolve_tool_call(&ctx(), tool_call_event()).await
    }

    #[tokio::test]
    async fn erased_dispatch_uses_the_public_on_tool_call_method() {
        let calls = Arc::new(AtomicUsize::new(0));
        let stack = HookStack::with(OnToolCallOnly(calls.clone()));

        let (action, salvaged) = resolve(&stack).await;

        assert_eq!(action, ToolCallAction::skip("called"));
        assert_eq!(salvaged, None);
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn string_rewrite_is_json_encoded_for_later_hook_in_same_stack() {
        let spy = ArgsSpy::default();
        let replacement = Value::String("sanitized".into());
        let mut stack = HookStack::new();
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
        let inner = HookStack::with(spy.clone());
        let mut outer = HookStack::new();
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
        let mut inner = HookStack::new();
        inner.push(RewriteHook(json!({"x":41})));
        inner.push(SkipHook);
        let mut outer = HookStack::new();
        outer.push(inner);
        let (action, salvaged) = resolve(&outer).await;
        assert!(matches!(action, ToolCallAction::Skip(_)));
        assert_eq!(salvaged, Some(json!({"x":41})));
    }

    #[tokio::test]
    async fn nested_rewrite_then_stop_preserves_rewrite() {
        let mut inner = HookStack::new();
        inner.push(RewriteHook(json!({"x":41})));
        inner.push(StopHook);
        let mut outer = HookStack::new();
        outer.push(inner);
        let (action, salvaged) = resolve(&outer).await;
        assert!(matches!(action, ToolCallAction::Stop(_)));
        assert_eq!(salvaged, Some(json!({"x":41})));
    }

    #[tokio::test]
    async fn deeply_nested_terminal_action_preserves_the_last_rewrite() {
        let mut inner = HookStack::new();
        inner.push(RewriteHook(json!({"x":3})));
        inner.push(SkipHook);

        let mut middle = HookStack::new();
        middle.push(RewriteHook(json!({"x":2})));
        middle.push(inner);

        let mut outer = HookStack::new();
        outer.push(RewriteHook(json!({"x":1})));
        outer.push(middle);

        let (action, salvaged) = resolve(&outer).await;

        assert_eq!(action, ToolCallAction::skip("denied"));
        assert_eq!(salvaged, Some(json!({"x":3})));
    }

    #[tokio::test]
    async fn concurrent_nested_resolutions_keep_rewrites_isolated_by_call() {
        let mut inner = HookStack::new();
        inner.push(YieldingRewriteFromCallId);
        inner.push(YieldingSkip);
        let outer = HookStack::with(inner);
        let context = ctx();

        let first_id = InternalCallId::new();
        let second_id = InternalCallId::new();
        let first = outer.resolve_tool_call(
            &context,
            ToolCall {
                internal_call_id: first_id,
                ..tool_call_event()
            },
        );
        let second = outer.resolve_tool_call(
            &context,
            ToolCall {
                internal_call_id: second_id,
                ..tool_call_event()
            },
        );
        let ((first_action, first_rewrite), (second_action, second_rewrite)) =
            tokio::join!(first, second);

        assert_eq!(first_action, ToolCallAction::skip("denied"));
        assert_eq!(first_rewrite, Some(json!({"call_id": first_id})));
        assert_eq!(second_action, ToolCallAction::skip("denied"));
        assert_eq!(second_rewrite, Some(json!({"call_id": second_id})));
    }

    #[tokio::test]
    async fn outer_rewrite_threads_into_nested_stack() {
        let spy = ArgsSpy::default();
        let mut inner = HookStack::new();
        inner.push(spy.clone());
        inner.push(SkipHook);
        let mut outer = HookStack::new();
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
        let mut proceed = HookStack::new();
        proceed.push(RewriteHook(json!({"x":5})));
        let (action, salvaged) = resolve(&proceed).await;
        assert_eq!(action, ToolCallAction::rewrite(json!({"x":5})));
        assert_eq!(salvaged, None);
    }

    #[test]
    fn action_types_are_event_specific() {
        fn model_selection(_: ModelSelectionAction) {}
        fn completion(_: CompletionCallAction) {}
        fn model_turn(_: ModelTurnAction) {}
        fn retry_request(_: RetryRequest) {}
        fn call(_: ToolCallAction) {}
        fn result(_: ToolResultAction) {}
        fn invalid(_: InvalidToolCallAction) {}
        fn observation(_: ObservationAction) {}
        model_selection(ModelSelectionAction::continue_run());
        completion(CompletionCallAction::continue_run());
        model_turn(ModelTurnAction::retry_with_feedback("try again"));
        retry_request(RetryRequest::Repeat);
        call(ToolCallAction::run());
        result(ToolResultAction::keep());
        invalid(InvalidToolCallAction::fail());
        observation(ObservationAction::continue_run());
        let calls = AtomicUsize::new(0);
        calls.fetch_add(1, Ordering::Relaxed);
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }
}
