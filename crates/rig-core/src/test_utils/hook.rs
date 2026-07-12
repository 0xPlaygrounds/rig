//! Public harness for deterministic hook-policy tests.

use crate::{
    agent::{AgentHook, Flow, HookContext, HookStack, StepEvent},
    completion::CompletionModel,
};

/// A reusable hook stack with a stable synthetic run context.
pub struct HookHarness<M>
where
    M: CompletionModel,
{
    context: HookContext,
    hooks: HookStack<M>,
}

impl<M> HookHarness<M>
where
    M: CompletionModel,
{
    /// Creates an empty harness.
    pub fn new(is_streaming: bool, agent_name: Option<String>) -> Self {
        Self {
            context: HookContext::with_run_id(
                crate::agent::RunId::generate(),
                is_streaming,
                agent_name,
            ),
            hooks: HookStack::new(),
        }
    }

    /// Appends a hook in registration order.
    pub fn push<H>(&mut self, hook: H)
    where
        H: AgentHook<M> + 'static,
    {
        self.hooks.push(hook);
    }

    /// Returns the stable context used by every dispatch.
    pub fn context(&self) -> &HookContext {
        &self.context
    }

    /// Dispatches an event through normal [`HookStack`] composition.
    pub async fn dispatch(&self, event: StepEvent<'_, M>) -> Flow {
        self.hooks.on_event(&self.context, event).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{agent::StepEvent, test_utils::MockCompletionModel};

    struct Stop;
    impl AgentHook<MockCompletionModel> for Stop {
        async fn on_event(
            &self,
            context: &HookContext,
            event: StepEvent<'_, MockCompletionModel>,
        ) -> Flow {
            if let StepEvent::ToolCall {
                internal_call_id, ..
            } = event
            {
                context.call_scratchpad(internal_call_id).insert::<u32>(7);
                Flow::terminate("blocked")
            } else {
                Flow::cont()
            }
        }
    }

    #[tokio::test]
    async fn harness_exercises_stack_and_call_scoped_state() {
        let mut harness = HookHarness::<MockCompletionModel>::new(false, Some("test".into()));
        harness.push(Stop);
        let flow = harness
            .dispatch(StepEvent::ToolCall {
                tool_name: "tool",
                tool_call_id: Some("provider"),
                internal_call_id: "internal",
                parent_internal_call_id: None,
                args: "{}",
            })
            .await;
        assert!(matches!(flow, Flow::Terminate { .. }));
        assert_eq!(
            harness.context().call_scratchpad("internal").get::<u32>(),
            Some(7)
        );
        assert!(
            harness
                .context()
                .call_scratchpad("other")
                .get::<u32>()
                .is_none()
        );
    }
}
