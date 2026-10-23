use rig_core::conversation::Aside;

use crate::agent::{AgentHook, HookContext, HookStack};
use crate::handlers::Handlers;

/// A hook stack and the run context its hooks see, for testing hooks
/// without running an agent. Call any [`AgentHook`] method on
/// [`hooks`](Self::hooks) with [`context`](Self::context); the context
/// keeps its scratchpad and entries across calls, as a run does. It has no
/// handlers, so a hook that dispatches through it needs an agent test.
///
/// ```
/// use rig_agent::agent::{AgentHook, HookContext, ObservationAction, TextDelta};
/// use rig_agent::test_utils::HookTestHarness;
///
/// struct CountDeltas;
/// impl AgentHook for CountDeltas {
///     async fn on_text_delta(&self, ctx: &HookContext, _: TextDelta<'_>) -> ObservationAction {
///         ctx.scratchpad().update(|count: &mut usize| *count += 1);
///         ObservationAction::Continue
///     }
/// }
///
/// # futures::executor::block_on(async {
/// let harness = HookTestHarness::new().with_hook(CountDeltas);
/// let delta = TextDelta { delta: "hi", aggregated: "hi" };
/// harness.hooks().on_text_delta(harness.context(), delta).await;
/// assert_eq!(harness.context().scratchpad().get::<usize>(), Some(1));
/// # });
/// ```
#[derive(Debug)]
pub struct HookTestHarness {
    hooks: HookStack,
    context: HookContext,
}

impl HookTestHarness {
    /// An empty stack and a fresh unary run at turn zero.
    pub fn new() -> Self {
        Self::with_context(false, None)
    }

    fn with_context(is_streaming: bool, agent_name: Option<String>) -> Self {
        Self {
            hooks: HookStack::new(),
            context: HookContext::new(is_streaming, agent_name, Handlers::default()),
        }
    }

    /// Append `hook`; hooks run in the order they were added.
    pub fn with_hook(mut self, hook: impl AgentHook + 'static) -> Self {
        self.hooks.push(hook);
        self
    }

    /// Make the run a streaming one, or not. Call it before seeding state:
    /// the context is rebuilt.
    pub fn with_streaming(self, is_streaming: bool) -> Self {
        let agent_name = self.context.agent_name().map(str::to_owned);
        self.rebuilt(is_streaming, agent_name)
    }

    /// Name the run's agent. Call it before seeding state: the context is
    /// rebuilt.
    pub fn with_agent_name(self, name: impl Into<String>) -> Self {
        let is_streaming = self.context.is_streaming();
        self.rebuilt(is_streaming, Some(name.into()))
    }

    fn rebuilt(self, is_streaming: bool, agent_name: Option<String>) -> Self {
        Self {
            hooks: self.hooks,
            ..Self::with_context(is_streaming, agent_name)
        }
    }

    /// Seed the entries a resumed run carries, in order.
    pub fn with_entries(self, entries: &[Aside]) -> Self {
        self.context.seed_entries(entries);
        self
    }

    /// Set the context's one-based model-call index; zero is before the
    /// first call. Run state is kept.
    pub fn set_turn(&self, turn: usize) {
        self.context.set_turn(turn);
    }

    /// The stack, to call hook methods on.
    pub fn hooks(&self) -> &HookStack {
        &self.hooks
    }

    /// The run context, to pass to hook methods and inspect.
    pub fn context(&self) -> &HookContext {
        &self.context
    }
}

impl Default for HookTestHarness {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
