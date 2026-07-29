//! Concrete, attach-and-forget hooks for the session drivers.
//!
//! [`Hooks`] is an ordered list of named callback records ([`HookEntry`])
//! dispatched over owned [`HookEvent`] values and folded into the existing
//! serde decision vocabulary ([`HookDecision`]). Fold semantics are exactly
//! those of the classic [`HookStack`](crate::agent::HookStack):
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
//! The folds reuse the shared composition helpers
//! ([`fold_completion_actions`], [`fold_observation_actions`],
//! [`fold_invalid_resolutions`], [`ToolCallResolution`],
//! [`ToolResultResolution`]) so this layer cannot drift from the classic
//! stack.
//!
//! A callback answers with the decision variant matching the event it
//! received; any other variant (including [`HookDecision::Continue`]) is
//! treated as "no opinion" for that event.

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
        /// The provider response.
        response: CompletionResponse,
    },
    /// Pre-execution decision point for one tool call. `call` carries the
    /// effective arguments, including rewrites from earlier entries. Answer
    /// with [`HookDecision::ToolCall`].
    ToolCall {
        /// The tool call about to execute.
        call: ToolCall,
    },
    /// Post-execution decision point for one tool result. `presentation`
    /// carries the running presentation rewrite from earlier entries;
    /// `result` always carries the raw execution result. Answer with
    /// [`HookDecision::ToolResult`].
    ToolResult {
        /// The executed tool call (with effective arguments).
        call: ToolCall,
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
    callback: Arc<dyn HookCallback>,
}

impl std::fmt::Debug for HookEntry {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("HookEntry")
            .field("name", &self.name)
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
            callback: Arc::new(callback),
        }
    }

    /// The entry's name.
    pub fn name(&self) -> &str {
        &self.name
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

    /// Dispatch [`HookEvent::BeforeModelCall`]: patches merge in
    /// registration order, the first `Stop` short-circuits (later entries
    /// are not invoked), mirroring `HookStack::on_completion_call` via
    /// [`fold_completion_actions`].
    pub async fn dispatch_completion_call(
        &self,
        turn: usize,
        prompt: &Message,
        history: &[Message],
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
    /// wins and later entries are not invoked, mirroring
    /// `HookStack::on_model_turn_finished`.
    pub async fn dispatch_model_turn(
        &self,
        turn: usize,
        content: &OneOrMany<AssistantContent>,
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
    /// observation wins, mirroring `HookStack::on_completion_response` via
    /// [`fold_observation_actions`].
    pub async fn dispatch_completion_response(
        &self,
        turn: usize,
        response: &CompletionResponse,
    ) -> ObservationAction {
        let mut actions = Vec::new();
        for entry in &self.entries {
            let decision = entry
                .dispatch(HookEvent::CompletionResponse {
                    turn,
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
    /// `HookStack::on_invalid_tool_call` via [`fold_invalid_resolutions`].
    pub async fn dispatch_invalid_tool_call(
        &self,
        context: &InvalidToolCallContext,
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
    /// mirroring `HookStack::resolve_tool_call` via [`ToolCallResolution`].
    ///
    /// Returns the effective action plus, for a terminal action, any
    /// rewrite salvaged before it (so the driver can report effective
    /// arguments).
    pub async fn dispatch_tool_call(
        &self,
        call: &ToolCall,
    ) -> (ToolCallAction, Option<serde_json::Value>) {
        let mut resolution = ToolCallResolution::new(call.function.arguments.clone());
        for entry in &self.entries {
            let mut current = call.clone();
            current.function.arguments = resolution.args().clone();
            let decision = entry.dispatch(HookEvent::ToolCall { call: current }).await;
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
    /// `HookStack::on_tool_result` via [`ToolResultResolution`].
    pub async fn dispatch_tool_result(
        &self,
        call: &ToolCall,
        result: &ToolResult,
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
    /// observation wins, mirroring `HookStack::on_stream_response_finish`
    /// via [`fold_observation_actions`].
    pub async fn dispatch_stream_finish(&self, final_record: &StreamFinal) -> ObservationAction {
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

    #[tokio::test]
    async fn invalid_resolution_first_some_wins() {
        let context = InvalidToolCallContext {
            tool_name: "missing".into(),
            tool_call_id: None,
            internal_call_id: None,
            args: None,
            available_tools: vec![],
            allowed_tools: vec![],
            tool_choice: None,
            chat_history: vec![],
            is_streaming: false,
        };
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
            if let HookEvent::ToolCall { call } = &event {
                observed
                    .lock()
                    .expect("lock")
                    .push(call.function.arguments.clone());
            }
            Box::pin(async { HookDecision::Continue })
        }));
        let (action, salvaged) = hooks
            .dispatch_tool_call(&tool_call(serde_json::json!({"a": 1})))
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
            .dispatch_tool_call(&tool_call(serde_json::json!({"a": 1})))
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
        let action = hooks.dispatch_tool_result(&call, &result).await;
        assert_eq!(action, ToolResultAction::rewrite("redacted"));
        assert_eq!(seen.lock().expect("lock").as_slice(), &["redacted"]);

        let mut hooks = Hooks::new();
        hooks.add(entry("stop", |_| {
            HookDecision::ToolResult(ToolResultAction::stop("leak"))
        }));
        hooks.add(entry("late", |_| {
            HookDecision::ToolResult(ToolResultAction::rewrite("never"))
        }));
        let action = hooks.dispatch_tool_result(&call, &result).await;
        assert_eq!(action, ToolResultAction::stop("leak"));
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
