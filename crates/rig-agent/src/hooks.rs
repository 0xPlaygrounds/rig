//! Concrete, attach-and-forget hooks for the session drivers.
//!
//! [`Hooks`] is an ordered list of named callback records ([`HookEntry`])
//! dispatched over owned [`HookEvent`] values and folded into the existing
//! serde decision vocabulary ([`HookDecision`]). Fold semantics are:
//!
//! - completion-call patches merge in registration order, the first `Stop`
//!   short-circuits (and later entries are not invoked);
//! - model-turn and observation events: the first non-`Continue` wins;
//! - invalid-call resolutions: the first `Some` wins (`None` everywhere
//!   preserves fail-fast behavior);
//! - tool-call argument rewrites and tool-result presentation rewrites
//!   chain into later entries, with a terminal `Skip`/`Stop` preserving the
//!   rewrite accumulated before it.
//!
//! The folds reuse the shared composition helpers in
//! [`crate::agent::hook`] ([`fold_completion_actions`],
//! [`fold_observation_actions`], [`fold_invalid_resolutions`],
//! [`ToolCallResolution`], [`ToolResultResolution`]) so every driver composes
//! decisions identically.
//!
//! A callback answers with the decision variant matching the event it
//! received; any other variant (including [`HookDecision::Continue`]) is
//! treated as "no opinion" for that event.
//!
//! # Delta events
//!
//! [`HookEvent::TextDelta`] and [`HookEvent::ToolCallDelta`] fire once per
//! streamed token, so they are opt-in: an entry receives them only when it
//! was built with [`HookEntry::observing_deltas`]. Drivers check
//! [`Hooks::observes_deltas`] once per run and skip building delta events
//! entirely when no entry opted in — the classic `StepEventKind`
//! interest hint, expressed as data.

use std::sync::Arc;

use rig_core::OneOrMany;
use rig_core::completion::CompletionResponse;
use rig_core::message::{AssistantContent, Message, ToolCall};
use rig_core::streaming::StreamFinal;
use rig_core::wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync};

use crate::agent::InvalidToolCallContext;
use crate::agent::hook::{
    CompletionCallAction, InvalidToolCallAction, ModelTurnAction, ObservationAction,
    ToolCallAction, ToolCallResolution, ToolResultAction, ToolResultResolution,
    fold_completion_actions, fold_invalid_resolutions, fold_observation_actions,
};
use crate::completion::Usage;
use crate::tool::{ToolOutput, ToolResult};

/// Owned hook events — the superset of decision points both session drivers
/// surface. Events carry owned data (no borrows) so callbacks can move them
/// across await points; payloads that are not serializable (for example
/// [`ToolResult`]) keep the enum itself non-serde by design.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum HookEvent {
    /// A model call is about to be prepared. Answer with
    /// [`HookDecision::CompletionCall`].
    BeforeModelCall {
        /// One-based model-call index.
        turn: usize,
        /// This turn's prompt message.
        prompt: Message,
        /// The history preceding it.
        history: Vec<Message>,
    },
    /// An accepted model turn awaits a verdict. Answer with
    /// [`HookDecision::ModelTurn`].
    ModelTurnFinished {
        /// One-based model-call index.
        turn: usize,
        /// Canonicalized assistant content parked for acceptance.
        content: OneOrMany<AssistantContent>,
        /// Usage reported for the turn.
        usage: Usage,
    },
    /// The full provider response for a completed turn. Answer with
    /// [`HookDecision::Observation`].
    CompletionResponse {
        /// One-based model-call index.
        turn: usize,
        /// The prompt sent for this turn.
        prompt: Message,
        /// The provider response.
        response: CompletionResponse,
    },
    /// Pre-execution decision point for one tool call. `call` carries the
    /// effective arguments, including rewrites from earlier entries. Answer
    /// with [`HookDecision::ToolCall`].
    ToolCall {
        /// The tool call about to execute.
        call: ToolCall,
        /// Rig correlation id for this call.
        internal_call_id: String,
    },
    /// Post-execution decision point for one tool result. `presentation`
    /// carries the running presentation rewrite from earlier entries;
    /// `result` always carries the raw execution result. Answer with
    /// [`HookDecision::ToolResult`].
    ToolResult {
        /// The executed tool call (with effective arguments).
        call: ToolCall,
        /// Rig correlation id for this call.
        internal_call_id: String,
        /// Immutable raw execution result.
        result: ToolResult,
        /// Current model-visible presentation, including earlier rewrites.
        presentation: ToolOutput,
    },
    /// The model emitted an unknown or disallowed tool call. Answer with
    /// [`HookDecision::InvalidToolCall`].
    InvalidToolCall(InvalidToolCallContext),
    /// The provider's terminal streaming record. Answer with
    /// [`HookDecision::Observation`].
    StreamFinish {
        /// The terminal stream record.
        final_record: StreamFinal,
    },
    /// A streamed turn's canonical response finished. Answer with
    /// [`HookDecision::Observation`].
    StreamResponseFinish {
        /// One-based model-call index.
        turn: usize,
        /// The prompt sent for this turn.
        prompt: Message,
        /// Canonical assistant content aggregated for this turn.
        content: OneOrMany<AssistantContent>,
        /// Usage reported for this turn.
        usage: Usage,
        /// Provider-assigned message id, when available.
        message_id: Option<String>,
    },
    /// One streamed text delta. Opt in with
    /// [`HookEntry::observing_deltas`]. Answer with
    /// [`HookDecision::Observation`].
    TextDelta {
        /// One-based model-call index.
        turn: usize,
        /// Newly received text.
        delta: String,
        /// Text accumulated for the turn so far.
        aggregated: String,
    },
    /// One streamed tool-call argument delta. Opt in with
    /// [`HookEntry::observing_deltas`]. Answer with
    /// [`HookDecision::Observation`].
    ToolCallDelta {
        /// One-based model-call index.
        turn: usize,
        /// Provider tool-call id.
        tool_call_id: String,
        /// Rig correlation id.
        internal_call_id: String,
        /// Tool name, on the first delta of a call.
        tool_name: Option<String>,
        /// Newly received argument fragment.
        delta: String,
    },
}

/// Owned decisions — a thin wrapper over the existing serde decision
/// vocabulary in [`crate::agent::hook`]. A variant that does not match the
/// dispatched event is treated as [`HookDecision::Continue`].
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum HookDecision {
    /// No opinion: defer to later entries / the event's default.
    Continue,
    /// Answer for [`HookEvent::BeforeModelCall`].
    CompletionCall(CompletionCallAction),
    /// Answer for [`HookEvent::ModelTurnFinished`].
    ModelTurn(ModelTurnAction),
    /// Answer for [`HookEvent::ToolCall`].
    ToolCall(ToolCallAction),
    /// Answer for [`HookEvent::ToolResult`].
    ToolResult(ToolResultAction),
    /// Answer for [`HookEvent::InvalidToolCall`].
    InvalidToolCall(InvalidToolCallAction),
    /// Answer for observation events ([`HookEvent::CompletionResponse`],
    /// [`HookEvent::StreamFinish`]).
    Observation(ObservationAction),
}

// Helper supertrait so the callback dyn is nameable (mirrors
// `PortableDynamicTool`'s `PortableDynamicCallback` pattern — WasmCompat
// bounds are not auto traits, so a plain `dyn Fn` alias cannot carry them).
trait HookCallback:
    Fn(HookEvent) -> WasmBoxedFuture<'static, HookDecision> + WasmCompatSend + WasmCompatSync
{
}

impl<F> HookCallback for F where
    F: Fn(HookEvent) -> WasmBoxedFuture<'static, HookDecision> + WasmCompatSend + WasmCompatSync
{
}

/// One named hook callback record.
#[derive(Clone)]
pub struct HookEntry {
    name: String,
    /// Whether this entry receives the high-frequency streamed delta events
    /// ([`HookEvent::TextDelta`], [`HookEvent::ToolCallDelta`]). `false` by
    /// default: the data form of the classic `observes(StepEventKind)` hint.
    observes_deltas: bool,
    callback: Arc<dyn HookCallback>,
}

impl std::fmt::Debug for HookEntry {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("HookEntry")
            .field("name", &self.name)
            .field("observes_deltas", &self.observes_deltas)
            .finish_non_exhaustive()
    }
}

impl HookEntry {
    /// Create a named hook entry from an owned async callback.
    pub fn new<F>(name: impl Into<String>, callback: F) -> Self
    where
        F: Fn(HookEvent) -> WasmBoxedFuture<'static, HookDecision>
            + WasmCompatSend
            + WasmCompatSync
            + 'static,
    {
        Self {
            name: name.into(),
            observes_deltas: false,
            callback: Arc::new(callback),
        }
    }

    /// The entry's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Opt this entry into the high-frequency streamed delta events
    /// ([`HookEvent::TextDelta`], [`HookEvent::ToolCallDelta`]). Entries that
    /// do not opt in never receive them, and a [`Hooks`] list where no entry
    /// opted in skips building them at all.
    pub fn observing_deltas(mut self) -> Self {
        self.observes_deltas = true;
        self
    }

    /// Whether this entry receives streamed delta events.
    pub fn observes_deltas(&self) -> bool {
        self.observes_deltas
    }

    async fn dispatch(&self, event: HookEvent) -> HookDecision {
        (self.callback)(event).await
    }
}

/// Ordered, attach-and-forget hook list. See the [module docs](self) for
/// the fold semantics.
#[derive(Debug, Default, Clone)]
pub struct Hooks {
    entries: Vec<HookEntry>,
}

impl Hooks {
    /// Creates an empty hook list.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends an entry to the end of the registration order.
    pub fn add(&mut self, entry: HookEntry) {
        self.entries.push(entry);
    }

    /// Builder-style [`Hooks::add`].
    pub fn with(mut self, entry: HookEntry) -> Self {
        self.add(entry);
        self
    }

    /// Whether the list contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The number of registered entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether any entry opted into the streamed delta events. Drivers check
    /// this once per run and skip building delta events entirely when it is
    /// `false`.
    pub fn observes_deltas(&self) -> bool {
        self.entries.iter().any(HookEntry::observes_deltas)
    }

    /// Dispatch [`HookEvent::BeforeModelCall`]: patches merge in
    /// registration order, the first `Stop` short-circuits (later entries
    /// are not invoked), via
    /// [`fold_completion_actions`].
    pub async fn dispatch_completion_call<'a>(
        &'a self,
        turn: usize,
        prompt: &'a Message,
        history: &'a [Message],
    ) -> CompletionCallAction {
        let mut actions = Vec::new();
        for entry in &self.entries {
            let decision = entry
                .dispatch(HookEvent::BeforeModelCall {
                    turn,
                    prompt: prompt.clone(),
                    history: history.to_vec(),
                })
                .await;
            let action = match decision {
                HookDecision::CompletionCall(action) => action,
                _ => CompletionCallAction::Continue,
            };
            let stop = matches!(action, CompletionCallAction::Stop(_));
            actions.push(action);
            if stop {
                break;
            }
        }
        fold_completion_actions(actions)
    }

    /// Dispatch [`HookEvent::ModelTurnFinished`]: the first non-`Continue`
    /// wins and later entries are not invoked.
    pub async fn dispatch_model_turn<'a>(
        &'a self,
        turn: usize,
        content: &'a OneOrMany<AssistantContent>,
        usage: Usage,
    ) -> ModelTurnAction {
        for entry in &self.entries {
            let decision = entry
                .dispatch(HookEvent::ModelTurnFinished {
                    turn,
                    content: content.clone(),
                    usage,
                })
                .await;
            if let HookDecision::ModelTurn(action) = decision
                && !matches!(action, ModelTurnAction::Continue)
            {
                return action;
            }
        }
        ModelTurnAction::Continue
    }

    /// Dispatch [`HookEvent::CompletionResponse`]: the first non-`Continue`
    /// observation wins, via
    /// [`fold_observation_actions`].
    pub async fn dispatch_completion_response<'a>(
        &'a self,
        turn: usize,
        prompt: &'a Message,
        response: &'a CompletionResponse,
    ) -> ObservationAction {
        let mut actions = Vec::new();
        for entry in &self.entries {
            let decision = entry
                .dispatch(HookEvent::CompletionResponse {
                    turn,
                    prompt: prompt.clone(),
                    response: response.clone(),
                })
                .await;
            let action = match decision {
                HookDecision::Observation(action) => action,
                _ => ObservationAction::Continue,
            };
            let stop = !matches!(action, ObservationAction::Continue);
            actions.push(action);
            if stop {
                break;
            }
        }
        fold_observation_actions(actions)
    }

    /// Dispatch [`HookEvent::InvalidToolCall`]: the first `Some` resolution
    /// wins (later entries are not invoked), mirroring
    /// via [`fold_invalid_resolutions`].
    pub async fn dispatch_invalid_tool_call<'a>(
        &'a self,
        context: &'a InvalidToolCallContext,
    ) -> Option<InvalidToolCallAction> {
        let mut resolutions = Vec::new();
        for entry in &self.entries {
            let decision = entry
                .dispatch(HookEvent::InvalidToolCall(context.clone()))
                .await;
            let resolution = match decision {
                HookDecision::InvalidToolCall(action) => Some(action),
                _ => None,
            };
            let resolved = resolution.is_some();
            resolutions.push(resolution);
            if resolved {
                break;
            }
        }
        fold_invalid_resolutions(resolutions)
    }

    /// Dispatch [`HookEvent::ToolCall`]: rewrites chain into later entries
    /// (each sees the current effective arguments), a terminal `Skip`/`Stop`
    /// short-circuits while preserving the rewrite accumulated before it,
    /// via [`ToolCallResolution`].
    ///
    /// Returns the effective action plus, for a terminal action, any
    /// rewrite salvaged before it (so the driver can report effective
    /// arguments).
    pub async fn dispatch_tool_call<'a>(
        &'a self,
        call: &'a ToolCall,
        internal_call_id: &'a str,
    ) -> (ToolCallAction, Option<serde_json::Value>) {
        let mut resolution = ToolCallResolution::new(call.function.arguments.clone());
        for entry in &self.entries {
            let mut current = call.clone();
            current.function.arguments = resolution.args().clone();
            let decision = entry
                .dispatch(HookEvent::ToolCall {
                    call: current,
                    internal_call_id: internal_call_id.to_owned(),
                })
                .await;
            let action = match decision {
                HookDecision::ToolCall(action) => action,
                _ => ToolCallAction::Run,
            };
            if !resolution.apply(action) {
                break;
            }
        }
        resolution.finish()
    }

    /// Dispatch [`HookEvent::ToolResult`]: presentation rewrites chain into
    /// later entries (each sees the current effective presentation; the raw
    /// result is unchanged), `Stop` short-circuits, mirroring
    /// via [`ToolResultResolution`].
    pub async fn dispatch_tool_result<'a>(
        &'a self,
        call: &'a ToolCall,
        internal_call_id: &'a str,
        result: &'a ToolResult,
    ) -> ToolResultAction {
        let mut resolution = ToolResultResolution::new();
        for entry in &self.entries {
            let presentation = resolution
                .presentation()
                .unwrap_or_else(|| result.output())
                .clone();
            let decision = entry
                .dispatch(HookEvent::ToolResult {
                    call: call.clone(),
                    internal_call_id: internal_call_id.to_owned(),
                    result: result.clone(),
                    presentation,
                })
                .await;
            let action = match decision {
                HookDecision::ToolResult(action) => action,
                _ => ToolResultAction::Keep,
            };
            if !resolution.apply(action) {
                break;
            }
        }
        resolution.finish()
    }

    /// Dispatch [`HookEvent::StreamFinish`]: the first non-`Continue`
    /// observation wins, via [`fold_observation_actions`].
    pub async fn dispatch_stream_finish<'a>(
        &'a self,
        final_record: &'a StreamFinal,
    ) -> ObservationAction {
        let mut actions = Vec::new();
        for entry in &self.entries {
            let decision = entry
                .dispatch(HookEvent::StreamFinish {
                    final_record: final_record.clone(),
                })
                .await;
            let action = match decision {
                HookDecision::Observation(action) => action,
                _ => ObservationAction::Continue,
            };
            let stop = !matches!(action, ObservationAction::Continue);
            actions.push(action);
            if stop {
                break;
            }
        }
        fold_observation_actions(actions)
    }

    /// Dispatch [`HookEvent::StreamResponseFinish`]: the first non-`Continue`
    /// observation wins, via [`fold_observation_actions`].
    pub async fn dispatch_stream_response_finish<'a>(
        &'a self,
        turn: usize,
        prompt: &'a Message,
        content: &'a OneOrMany<AssistantContent>,
        usage: Usage,
        message_id: Option<&'a str>,
    ) -> ObservationAction {
        self.fold_observations(|| HookEvent::StreamResponseFinish {
            turn,
            prompt: prompt.clone(),
            content: content.clone(),
            usage,
            message_id: message_id.map(str::to_owned),
        })
        .await
    }

    /// Dispatch [`HookEvent::TextDelta`] to the entries that opted into delta
    /// observation: the first non-`Continue` observation wins.
    pub async fn dispatch_text_delta<'a>(
        &'a self,
        turn: usize,
        delta: &'a str,
        aggregated: &'a str,
    ) -> ObservationAction {
        self.fold_delta_observations(|| HookEvent::TextDelta {
            turn,
            delta: delta.to_owned(),
            aggregated: aggregated.to_owned(),
        })
        .await
    }

    /// Dispatch [`HookEvent::ToolCallDelta`] to the entries that opted into
    /// delta observation: the first non-`Continue` observation wins.
    pub async fn dispatch_tool_call_delta<'a>(
        &'a self,
        turn: usize,
        tool_call_id: &'a str,
        internal_call_id: &'a str,
        tool_name: Option<&'a str>,
        delta: &'a str,
    ) -> ObservationAction {
        self.fold_delta_observations(|| HookEvent::ToolCallDelta {
            turn,
            tool_call_id: tool_call_id.to_owned(),
            internal_call_id: internal_call_id.to_owned(),
            tool_name: tool_name.map(str::to_owned),
            delta: delta.to_owned(),
        })
        .await
    }

    /// Shared observation fold: build the event per entry (events are owned,
    /// so each entry gets its own copy), stop invoking after the first
    /// non-`Continue`.
    async fn fold_observations(&self, event: impl Fn() -> HookEvent) -> ObservationAction {
        let mut actions = Vec::new();
        for entry in &self.entries {
            let decision = entry.dispatch(event()).await;
            let action = match decision {
                HookDecision::Observation(action) => action,
                _ => ObservationAction::Continue,
            };
            let stop = !matches!(action, ObservationAction::Continue);
            actions.push(action);
            if stop {
                break;
            }
        }
        fold_observation_actions(actions)
    }

    /// [`Self::fold_observations`] restricted to delta-observing entries.
    async fn fold_delta_observations(&self, event: impl Fn() -> HookEvent) -> ObservationAction {
        let mut actions = Vec::new();
        for entry in self.entries.iter().filter(|entry| entry.observes_deltas) {
            let decision = entry.dispatch(event()).await;
            let action = match decision {
                HookDecision::Observation(action) => action,
                _ => ObservationAction::Continue,
            };
            let stop = !matches!(action, ObservationAction::Continue);
            actions.push(action);
            if stop {
                break;
            }
        }
        fold_observation_actions(actions)
    }
}

/// The agent data model is plain `Send + Sync` data end to end — the
/// property every driver future's `Send`-ness rests on. A regression here
/// (a non-`Send` slot sneaking into a hook record, executor, agent, or
/// session) fails to compile right here instead of at some distant
/// `async_trait` call site.
#[allow(dead_code)]
fn _assert_agent_model_is_send_and_sync() {
    fn assert<T: Send + Sync>() {}
    assert::<Hooks>();
    assert::<HookEntry>();
    assert::<crate::executor::ToolExecutor>();
    assert::<crate::agent::Agent>();
    assert::<crate::session::AgentSession>();
    // `AgentStream` holds a boxed provider stream (a transport-edge `dyn`
    // that is `Send` but not `Sync`), so only `Send` is asserted for it.
    fn assert_send<T: Send>() {}
    assert_send::<crate::stream::AgentStream>();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::RequestPatch;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn entry(
        name: &str,
        decide: impl Fn(HookEvent) -> HookDecision + Send + Sync + 'static,
    ) -> HookEntry {
        HookEntry::new(name, move |event| {
            let decision = decide(event);
            Box::pin(async move { decision })
        })
    }

    fn tool_call(args: serde_json::Value) -> ToolCall {
        ToolCall::new(
            "call_1".to_string(),
            rig_core::message::ToolFunction {
                name: "add".to_string(),
                arguments: args,
            },
        )
    }

    #[tokio::test]
    async fn completion_call_merges_patches_in_order_and_stops_first() {
        let mut hooks = Hooks::new();
        hooks.add(entry("a", |_| {
            HookDecision::CompletionCall(CompletionCallAction::patch(
                RequestPatch::new().temperature(0.1),
            ))
        }));
        hooks.add(entry("b", |_| HookDecision::Continue));
        hooks.add(entry("c", |_| {
            HookDecision::CompletionCall(CompletionCallAction::patch(
                RequestPatch::new().temperature(0.2).max_tokens(9),
            ))
        }));
        let prompt = Message::from("hi");
        let folded = hooks.dispatch_completion_call(1, &prompt, &[]).await;
        assert_eq!(
            folded,
            CompletionCallAction::Patch(RequestPatch::new().temperature(0.2).max_tokens(9))
        );

        // First Stop wins and later entries are not invoked.
        let invoked = Arc::new(AtomicUsize::new(0));
        let counter = invoked.clone();
        let mut hooks = Hooks::new();
        hooks.add(entry("stop", |_| {
            HookDecision::CompletionCall(CompletionCallAction::stop("halt"))
        }));
        hooks.add(HookEntry::new("late", move |_| {
            counter.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { HookDecision::Continue })
        }));
        let stopped = hooks.dispatch_completion_call(1, &prompt, &[]).await;
        assert_eq!(stopped, CompletionCallAction::stop("halt"));
        assert_eq!(invoked.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn model_turn_first_non_continue_wins() {
        let mut hooks = Hooks::new();
        hooks.add(entry("observer", |_| HookDecision::Continue));
        hooks.add(entry("stopper", |_| {
            HookDecision::ModelTurn(ModelTurnAction::stop("done"))
        }));
        hooks.add(entry("late", |_| {
            HookDecision::ModelTurn(ModelTurnAction::repeat())
        }));
        let content = OneOrMany::one(AssistantContent::text("hi"));
        let action = hooks.dispatch_model_turn(1, &content, Usage::new()).await;
        assert_eq!(action, ModelTurnAction::stop("done"));
    }

    fn invalid_context() -> InvalidToolCallContext {
        InvalidToolCallContext {
            tool_name: "missing".into(),
            tool_call_id: None,
            internal_call_id: None,
            args: None,
            available_tools: vec![],
            allowed_tools: vec![],
            tool_choice: None,
            chat_history: vec![],
            is_streaming: false,
        }
    }

    #[tokio::test]
    async fn invalid_resolution_first_some_wins() {
        let context = invalid_context();
        let hooks = Hooks::new();
        assert_eq!(hooks.dispatch_invalid_tool_call(&context).await, None);

        let mut hooks = Hooks::new();
        hooks.add(entry("pass", |_| HookDecision::Continue));
        hooks.add(entry("skip", |_| {
            HookDecision::InvalidToolCall(InvalidToolCallAction::skip("nope"))
        }));
        hooks.add(entry("late", |_| {
            HookDecision::InvalidToolCall(InvalidToolCallAction::fail())
        }));
        assert_eq!(
            hooks.dispatch_invalid_tool_call(&context).await,
            Some(InvalidToolCallAction::skip("nope"))
        );
    }

    #[tokio::test]
    async fn tool_call_rewrites_chain_into_later_entries() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut hooks = Hooks::new();
        hooks.add(entry("rewrite", |_| {
            HookDecision::ToolCall(ToolCallAction::rewrite(serde_json::json!({"a": 2})))
        }));
        let observed = seen.clone();
        hooks.add(HookEntry::new("observer", move |event| {
            if let HookEvent::ToolCall { call, .. } = &event {
                observed
                    .lock()
                    .expect("lock")
                    .push(call.function.arguments.clone());
            }
            Box::pin(async { HookDecision::Continue })
        }));
        let (action, salvaged) = hooks
            .dispatch_tool_call(&tool_call(serde_json::json!({"a": 1})), "internal-1")
            .await;
        assert_eq!(action, ToolCallAction::Rewrite(serde_json::json!({"a": 2})));
        assert!(salvaged.is_none());
        // The later entry saw the rewritten arguments.
        assert_eq!(
            seen.lock().expect("lock").as_slice(),
            &[serde_json::json!({"a": 2})]
        );
    }

    #[tokio::test]
    async fn tool_call_terminal_skip_salvages_accumulated_rewrite() {
        let mut hooks = Hooks::new();
        hooks.add(entry("rewrite", |_| {
            HookDecision::ToolCall(ToolCallAction::rewrite(serde_json::json!({"a": 3})))
        }));
        hooks.add(entry("skip", |_| {
            HookDecision::ToolCall(ToolCallAction::skip("blocked"))
        }));
        hooks.add(entry("late", |_| {
            HookDecision::ToolCall(ToolCallAction::rewrite(serde_json::json!({"a": 9})))
        }));
        let (action, salvaged) = hooks
            .dispatch_tool_call(&tool_call(serde_json::json!({"a": 1})), "internal-1")
            .await;
        assert_eq!(action, ToolCallAction::skip("blocked"));
        assert_eq!(salvaged, Some(serde_json::json!({"a": 3})));
    }

    #[tokio::test]
    async fn tool_result_rewrites_chain_and_stop_short_circuits() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut hooks = Hooks::new();
        hooks.add(entry("rewrite", |_| {
            HookDecision::ToolResult(ToolResultAction::rewrite("redacted"))
        }));
        let observed = seen.clone();
        hooks.add(HookEntry::new("observer", move |event| {
            if let HookEvent::ToolResult { presentation, .. } = &event {
                observed.lock().expect("lock").push(presentation.render());
            }
            Box::pin(async { HookDecision::Continue })
        }));
        let call = tool_call(serde_json::json!({"a": 1}));
        let result = ToolResult::success(ToolOutput::text("raw"));
        let action = hooks
            .dispatch_tool_result(&call, "internal-1", &result)
            .await;
        assert_eq!(action, ToolResultAction::rewrite("redacted"));
        assert_eq!(seen.lock().expect("lock").as_slice(), &["redacted"]);
        // The raw result is never mutated by a presentation rewrite.
        assert_eq!(result.output().as_text(), Some("raw"));

        // A `Stop` short-circuits: the later entry is never invoked.
        let invoked = Arc::new(AtomicUsize::new(0));
        let counter = invoked.clone();
        let mut hooks = Hooks::new();
        hooks.add(entry("stop", |_| {
            HookDecision::ToolResult(ToolResultAction::stop("leak"))
        }));
        hooks.add(HookEntry::new("late", move |_| {
            counter.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { HookDecision::ToolResult(ToolResultAction::rewrite("never")) })
        }));
        let action = hooks
            .dispatch_tool_result(&call, "internal-1", &result)
            .await;
        assert_eq!(action, ToolResultAction::stop("leak"));
        assert_eq!(invoked.load(Ordering::SeqCst), 0);
    }

    /// An entry that appends `label` to `log` and answers `decide`.
    fn logging_entry(
        label: u32,
        log: Arc<Mutex<Vec<u32>>>,
        decide: impl Fn(HookEvent) -> HookDecision + Send + Sync + 'static,
    ) -> HookEntry {
        HookEntry::new(format!("entry-{label}"), move |event| {
            log.lock().expect("log").push(label);
            let decision = decide(event);
            Box::pin(async move { decision })
        })
    }

    // ── Migrated from the deleted `HookStack` suite: the ordered-dispatch
    // invariants, now asserted against `Hooks`. Each keeps the original's
    // invocation-log assertion, which is what proves the short-circuit
    // (a folded action alone cannot).

    #[tokio::test]
    async fn runs_entries_in_registration_order_and_consults_all_on_continue() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut hooks = Hooks::new();
        for label in [1, 2] {
            hooks.add(logging_entry(label, log.clone(), |_| {
                HookDecision::ToolCall(ToolCallAction::run())
            }));
        }
        let (action, salvaged) = hooks
            .dispatch_tool_call(&tool_call(serde_json::json!({})), "internal-1")
            .await;
        assert_eq!(action, ToolCallAction::run());
        assert_eq!(salvaged, None);
        assert_eq!(*log.lock().expect("log"), vec![1, 2]);
    }

    #[tokio::test]
    async fn first_stop_short_circuits_on_chained_tool_call() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut hooks = Hooks::new();
        hooks.add(logging_entry(1, log.clone(), |_| {
            HookDecision::ToolCall(ToolCallAction::stop("stop"))
        }));
        hooks.add(logging_entry(2, log.clone(), |_| {
            HookDecision::ToolCall(ToolCallAction::run())
        }));
        let (action, _) = hooks
            .dispatch_tool_call(&tool_call(serde_json::json!({})), "internal-1")
            .await;
        assert!(matches!(action, ToolCallAction::Stop(_)));
        assert_eq!(*log.lock().expect("log"), vec![1], "entry 2 must not run");
    }

    #[tokio::test]
    async fn first_stop_short_circuits_observation() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut hooks = Hooks::new();
        hooks.add(
            logging_entry(1, log.clone(), |_| {
                HookDecision::Observation(ObservationAction::stop("stop"))
            })
            .observing_deltas(),
        );
        hooks.add(
            logging_entry(2, log.clone(), |_| {
                HookDecision::Observation(ObservationAction::continue_run())
            })
            .observing_deltas(),
        );
        assert!(matches!(
            hooks.dispatch_text_delta(1, "hi", "hi").await,
            ObservationAction::Stop(_)
        ));
        assert_eq!(*log.lock().expect("log"), vec![1], "entry 2 must not run");
    }

    #[tokio::test]
    async fn explicit_fail_short_circuits_later_invalid_tool_entries() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut hooks = Hooks::new();
        hooks.add(logging_entry(1, log.clone(), |_| {
            HookDecision::InvalidToolCall(InvalidToolCallAction::fail())
        }));
        hooks.add(logging_entry(2, log.clone(), |_| {
            HookDecision::InvalidToolCall(InvalidToolCallAction::retry("try another tool"))
        }));
        assert_eq!(
            hooks.dispatch_invalid_tool_call(&invalid_context()).await,
            Some(InvalidToolCallAction::fail())
        );
        assert_eq!(*log.lock().expect("log"), vec![1], "entry 2 must not run");
    }

    #[tokio::test]
    async fn no_invalid_tool_decision_defers_to_later_entries() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut hooks = Hooks::new();
        // `Continue` is the data form of the classic hook's `None`.
        hooks.add(logging_entry(1, log.clone(), |_| HookDecision::Continue));
        hooks.add(logging_entry(2, log.clone(), |_| {
            HookDecision::InvalidToolCall(InvalidToolCallAction::retry("try another tool"))
        }));
        assert_eq!(
            hooks.dispatch_invalid_tool_call(&invalid_context()).await,
            Some(InvalidToolCallAction::retry("try another tool"))
        );
        assert_eq!(*log.lock().expect("log"), vec![1, 2]);
    }

    #[tokio::test]
    async fn completion_patches_accumulate_and_stop_discards_prior_patch() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let prompt = Message::from("hi");

        let mut hooks = Hooks::new();
        hooks.add(logging_entry(1, log.clone(), |_| {
            HookDecision::CompletionCall(CompletionCallAction::patch(
                RequestPatch::new().temperature(0.1),
            ))
        }));
        hooks.add(logging_entry(2, log.clone(), |_| {
            HookDecision::CompletionCall(CompletionCallAction::patch(
                RequestPatch::new().max_tokens(256),
            ))
        }));
        match hooks.dispatch_completion_call(1, &prompt, &[]).await {
            CompletionCallAction::Patch(patch) => {
                assert_eq!(patch.temperature, Some(0.1));
                assert_eq!(patch.max_tokens, Some(256));
            }
            other => panic!("expected a merged patch, got {other:?}"),
        }
        assert_eq!(*log.lock().expect("log"), vec![1, 2]);

        // A `Stop` wins outright and discards the patch accumulated before it.
        let mut hooks = Hooks::new();
        hooks.add(logging_entry(3, log.clone(), |_| {
            HookDecision::CompletionCall(CompletionCallAction::stop("stop"))
        }));
        hooks.add(logging_entry(4, log.clone(), |_| {
            HookDecision::CompletionCall(CompletionCallAction::patch(RequestPatch::new()))
        }));
        assert!(matches!(
            hooks.dispatch_completion_call(1, &prompt, &[]).await,
            CompletionCallAction::Stop(_)
        ));
        assert_eq!(
            *log.lock().expect("log"),
            vec![1, 2, 3],
            "entry 4 must not run"
        );
    }

    #[tokio::test]
    async fn delta_events_reach_only_opted_in_entries() {
        let plain_seen = Arc::new(AtomicUsize::new(0));
        let delta_seen = Arc::new(AtomicUsize::new(0));

        let mut hooks = Hooks::new();
        assert!(!hooks.observes_deltas());

        let plain = plain_seen.clone();
        hooks.add(HookEntry::new("plain", move |_| {
            plain.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { HookDecision::Continue })
        }));
        assert!(!hooks.observes_deltas());

        let observer = delta_seen.clone();
        hooks.add(
            HookEntry::new("observer", move |event| {
                if matches!(
                    event,
                    HookEvent::TextDelta { .. } | HookEvent::ToolCallDelta { .. }
                ) {
                    observer.fetch_add(1, Ordering::SeqCst);
                }
                Box::pin(async { HookDecision::Continue })
            })
            .observing_deltas(),
        );
        assert!(hooks.observes_deltas());

        assert_eq!(
            hooks.dispatch_text_delta(1, "hi", "hi").await,
            ObservationAction::Continue
        );
        assert_eq!(
            hooks
                .dispatch_tool_call_delta(1, "call_1", "internal-1", Some("add"), "{")
                .await,
            ObservationAction::Continue
        );
        // The non-observing entry was never invoked for either delta.
        assert_eq!(plain_seen.load(Ordering::SeqCst), 0);
        assert_eq!(delta_seen.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn delta_observation_stop_short_circuits() {
        let mut hooks = Hooks::new();
        hooks.add(entry("stop", |_| {
            HookDecision::Observation(ObservationAction::stop("enough"))
        }));
        let late = Arc::new(AtomicUsize::new(0));
        let counter = late.clone();
        hooks.add(
            HookEntry::new("late", move |_| {
                counter.fetch_add(1, Ordering::SeqCst);
                Box::pin(async { HookDecision::Continue })
            })
            .observing_deltas(),
        );
        // The stopping entry did not opt in, so it never sees the delta and the
        // opted-in entry does run.
        assert_eq!(
            hooks.dispatch_text_delta(1, "hi", "hi").await,
            ObservationAction::Continue
        );
        assert_eq!(late.load(Ordering::SeqCst), 1);

        // With both opted in, registration order plus first-stop-wins holds.
        let mut hooks = Hooks::new();
        hooks.add(
            entry("stop", |_| {
                HookDecision::Observation(ObservationAction::stop("enough"))
            })
            .observing_deltas(),
        );
        let counter = Arc::new(AtomicUsize::new(0));
        let seen = counter.clone();
        hooks.add(
            HookEntry::new("late", move |_| {
                seen.fetch_add(1, Ordering::SeqCst);
                Box::pin(async { HookDecision::Continue })
            })
            .observing_deltas(),
        );
        assert_eq!(
            hooks.dispatch_text_delta(1, "hi", "hi").await,
            ObservationAction::stop("enough")
        );
        assert_eq!(counter.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn mismatched_decision_variants_are_treated_as_continue() {
        let mut hooks = Hooks::new();
        // A tool-call answer to a model-turn event is ignored.
        hooks.add(entry("wrong", |_| {
            HookDecision::ToolCall(ToolCallAction::stop("wrong lane"))
        }));
        let content = OneOrMany::one(AssistantContent::text("hi"));
        assert_eq!(
            hooks.dispatch_model_turn(1, &content, Usage::new()).await,
            ModelTurnAction::Continue
        );
    }
}
