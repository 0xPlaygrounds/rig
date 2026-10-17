use super::{
    AgentHook, CompletionCallAction, CompletionCallEvent, DispatchAction, DispatchEvent,
    HookContext, HookStack, InvalidToolCallAction, InvalidToolCallContext, ModelSelection,
    ModelSelectionAction, ModelTurnAction, ModelTurnFinished, ObservationAction, OutcomeAction,
    OutcomeEvent, ReasoningDelta, RunEntry, RunId, RunSettled, RunStart, RunStartAction, TextDelta,
    ToolCallDelta,
};

/// Invokes an ordered hook stack against a persistent, isolated run context.
/// Callbacks return the stack's typed actions without executing an agent, model,
/// or tool. The context has no bus; hooks that dispatch work need a runner test.
///
/// Shared borrows allow concurrent callbacks. Event turns are caller-supplied;
/// set the context turn separately before dispatching events for that turn.
/// Explicit calls bypass the runner's aggregate observation hints; dispatch
/// and outcome callbacks retain the stack's per-hook interest filtering.
///
/// ```
/// use rig_agent::{
///     agent::{AgentHook, HookContext, ObservationAction, TextDelta},
///     test_utils::HookTestHarness,
/// };
///
/// struct CountDeltas;
/// impl AgentHook for CountDeltas {
///     async fn on_text_delta(&self, ctx: &HookContext, _: TextDelta<'_>) -> ObservationAction {
///         ctx.scratchpad().update::<usize, _>(|count| *count += 1);
///         ObservationAction::Continue
///     }
/// }
///
/// # futures::executor::block_on(async {
/// let harness = HookTestHarness::new().with_hook(CountDeltas).with_turn(1);
/// let action = harness.on_text_delta(TextDelta { delta: "hello", aggregated: "hello" }).await;
/// assert_eq!(action, ObservationAction::Continue);
/// assert_eq!(harness.context().scratchpad().get::<usize>(), Some(1));
/// # });
/// ```
#[derive(Debug)]
pub struct HookTestHarness {
    hooks: HookStack,
    context: HookContext,
}

impl Default for HookTestHarness {
    fn default() -> Self {
        Self {
            hooks: HookStack::new(),
            context: HookContext::new(false, None, None),
        }
    }
}

impl HookTestHarness {
    /// Create an empty stack with a fresh run ID, turn zero, and empty state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a hook in registration order.
    pub fn with_hook<H: AgentHook + 'static>(mut self, hook: H) -> Self {
        self.hooks.push(hook);
        self
    }

    /// Append an existing stack, preserving its ordering and nested semantics.
    pub fn with_hooks(self, hooks: HookStack) -> Self {
        self.with_hook(hooks)
    }

    /// Use a caller-supplied run ID.
    pub fn with_run_id(mut self, run_id: RunId) -> Self {
        self.context.run_id = run_id;
        self
    }

    /// Set the agent name visible to hooks.
    pub fn with_agent_name(mut self, name: impl Into<String>) -> Self {
        self.context.agent_name = Some(name.into());
        self
    }

    /// Set whether hooks see a streaming run.
    pub fn with_streaming(mut self, is_streaming: bool) -> Self {
        self.context.is_streaming = is_streaming;
        self
    }

    /// Set the context's model-call index; zero represents a run before its first call.
    pub fn with_turn(mut self, turn: usize) -> Self {
        self.set_turn(turn);
        self
    }

    /// Seed persisted entries in append order, as when resuming a run.
    pub fn with_entries(self, entries: &[RunEntry]) -> Self {
        self.context.seed_entries(entries);
        self
    }

    /// Change the context turn between callbacks without clearing run state.
    /// Event fields are unaffected.
    pub fn set_turn(&mut self, turn: usize) {
        self.context.set_turn(turn);
    }

    /// Inspect or seed the run scratchpad and inspect appended entries.
    pub fn context(&self) -> &HookContext {
        &self.context
    }

    /// Inspect the registered stack or invoke its trait methods directly.
    pub fn hooks(&self) -> &HookStack {
        &self.hooks
    }

    /// Invoke run-start hooks and return their composed action.
    pub async fn on_run_start(&self, event: RunStart<'_>) -> RunStartAction {
        self.hooks.on_run_start(&self.context, event).await
    }

    /// Invoke settlement observers without transferring entries into a run.
    pub async fn on_run_settled(&self, event: RunSettled<'_>) {
        self.hooks.on_run_settled(&self.context, event).await;
    }

    /// Invoke synchronous model selection and return the composed action.
    pub fn on_model_select(&self, event: ModelSelection<'_>) -> ModelSelectionAction {
        self.hooks.on_model_select(&self.context, event)
    }

    /// Invoke completion-call hooks and return their merged patch or stop.
    pub async fn on_completion_call(&self, event: CompletionCallEvent<'_>) -> CompletionCallAction {
        self.hooks.on_completion_call(&self.context, event).await
    }

    /// Invoke model-turn hooks and return their first non-continuing action.
    pub async fn on_model_turn_finished(&self, event: ModelTurnFinished<'_>) -> ModelTurnAction {
        self.hooks
            .on_model_turn_finished(&self.context, event)
            .await
    }

    /// Invoke invalid-call hooks and return the first supplied policy action.
    pub async fn on_invalid_tool_call(
        &self,
        event: &InvalidToolCallContext,
    ) -> Option<InvalidToolCallAction> {
        self.hooks.on_invalid_tool_call(&self.context, event).await
    }

    /// Invoke text observers and return their first non-continuing action.
    pub async fn on_text_delta(&self, event: TextDelta<'_>) -> ObservationAction {
        self.hooks.on_text_delta(&self.context, event).await
    }

    /// Invoke reasoning observers and return their first non-continuing action.
    pub async fn on_reasoning_delta(&self, event: ReasoningDelta<'_>) -> ObservationAction {
        self.hooks.on_reasoning_delta(&self.context, event).await
    }

    /// Invoke tool-argument observers and return their first non-continuing action.
    pub async fn on_tool_call_delta(&self, event: ToolCallDelta<'_>) -> ObservationAction {
        self.hooks.on_tool_call_delta(&self.context, event).await
    }

    /// Invoke interested dispatch hooks and return a validated patch or denial.
    pub async fn on_dispatch(&self, event: DispatchEvent<'_>) -> DispatchAction {
        self.hooks.on_dispatch(&self.context, event).await
    }

    /// Invoke interested outcome hooks and return their composed replacement.
    pub async fn on_outcome(&self, event: OutcomeEvent<'_>) -> OutcomeAction {
        self.hooks.on_outcome(&self.context, event).await
    }
}
