//! Public harness for focused hook policy tests.

use crate::{
    agent::{AgentHook, Flow, HookContext, HookStack, RunContext, RunControlHandle, StepEvent},
    completion::CompletionModel,
};

/// Production-equivalent hook context plus an ordered hook stack.
pub struct HookTestHarness<M>
where
    M: CompletionModel,
{
    hooks: HookStack<M>,
    context: HookContext,
}

impl<M> HookTestHarness<M>
where
    M: CompletionModel,
{
    /// Build a non-streaming harness with a fresh run identity.
    pub fn new(hooks: HookStack<M>) -> Self {
        Self {
            hooks,
            context: HookContext::new(
                RunContext::new(RunControlHandle::new(), None),
                false,
                Some("hook-test".to_owned()),
            ),
        }
    }

    /// Select streaming metadata and an optional conversation identity.
    pub fn with_run_metadata(
        hooks: HookStack<M>,
        streaming: bool,
        conversation_id: Option<String>,
    ) -> Self {
        Self {
            hooks,
            context: HookContext::new(
                RunContext::new(RunControlHandle::new(), conversation_id),
                streaming,
                Some("hook-test".to_owned()),
            ),
        }
    }

    /// Set the one-based current turn.
    pub fn set_turn(&self, turn: usize) {
        self.context.set_turn(turn);
    }

    /// Dispatch one event through normal stack ordering semantics.
    pub async fn dispatch(&self, event: StepEvent<'_, M>) -> Flow {
        self.hooks.on_event(&self.context, event).await
    }

    /// Inspect run metadata and scratchpad state.
    pub fn context(&self) -> &HookContext {
        &self.context
    }
}
