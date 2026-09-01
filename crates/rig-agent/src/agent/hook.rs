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

pub use crate::run::patch::RequestPatch;

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
mod tests;

#[cfg(test)]
mod migrated_tests;
