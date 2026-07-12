use crate::{
    agent::Agent,
    completion::{CompletionModel, Prompt, PromptError},
    tool::{Tool, ToolCallExtensions},
};
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AgentToolArgs {
    /// The prompt for the agent to call.
    prompt: String,
}

impl<M: CompletionModel + 'static> Tool for Agent<M> {
    const NAME: &'static str = "agent_tool";

    type Error = PromptError;
    type Args = AgentToolArgs;
    type Output = String;

    fn description(&self) -> String {
        format!(
            "
            Prompt a sub-agent to do a task for you.

            Agent name: {name}
            Agent description: {description}
            Agent system prompt: {sysprompt}
            ",
            name = self.name(),
            description = self.description.clone().unwrap_or_default(),
            sysprompt = self.preamble.clone().unwrap_or_default()
        )
    }

    fn parameters(&self) -> serde_json::Value {
        json!(schema_for!(AgentToolArgs))
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.prompt(args.prompt).await
    }

    /// Propagate the caller's [`ToolCallExtensions`] into the sub-agent run, so the
    /// inner agent's own tools observe them too (sub-agent delegation / A2A
    /// chains). Without this, a sub-agent invoked as a tool would start with
    /// empty extensions.
    async fn call_with_extensions(
        &self,
        args: Self::Args,
        extensions: &ToolCallExtensions,
    ) -> Result<Self::Output, Self::Error> {
        let mut request = self.prompt(args.prompt).tool_extensions(extensions.clone());
        if let Some(context) = extensions.get::<crate::runtime::RunContext>() {
            let child_control = crate::runtime::RunControlHandle::for_child_context(context);
            request.runner = request
                .runner
                .inherit_runtime(child_control, context.clone());
        }
        request.await
    }

    fn name(&self) -> String {
        self.name.clone().unwrap_or_else(|| Self::NAME.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::AgentBuilder;
    use crate::runtime::RunContext;
    use crate::test_utils::{MockCompletionModel, MockExtensionsProbeTool, MockTurn, SessionId};
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    struct RuntimeProbe(Arc<Mutex<Option<RunContext>>>);

    impl Tool for RuntimeProbe {
        const NAME: &'static str = "runtime_probe";
        type Error = std::convert::Infallible;
        type Args = serde_json::Value;
        type Output = String;

        fn description(&self) -> String {
            "observe runtime context".into()
        }
        fn parameters(&self) -> serde_json::Value {
            json!({"type": "object"})
        }

        async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
            Ok("missing".into())
        }

        async fn call_with_extensions(
            &self,
            _args: Self::Args,
            extensions: &ToolCallExtensions,
        ) -> Result<Self::Output, Self::Error> {
            let context = extensions.get::<RunContext>().cloned();
            *self.0.lock().unwrap_or_else(|e| e.into_inner()) = context;
            Ok("observed".into())
        }
    }

    /// A `ToolCallExtensions` set on the outer run propagates into a sub-agent
    /// invoked as a tool, so the inner agent's own tools observe it.
    #[tokio::test]
    async fn context_propagates_into_sub_agent() {
        // Inner agent: calls a context-probing tool, then answers.
        let probe = MockExtensionsProbeTool::default();
        let inner_model = MockCompletionModel::new([
            MockTurn::tool_call("c1", "context_probe", json!({})),
            MockTurn::text("inner done"),
        ]);
        let inner = AgentBuilder::new(inner_model)
            .name("researcher")
            .tool(probe.clone())
            .build();

        // Outer agent: delegates to the inner agent (registered as the
        // "researcher" tool), then answers.
        let outer_model = MockCompletionModel::new([
            MockTurn::tool_call("c2", "researcher", json!({"prompt": "do research"})),
            MockTurn::text("outer done"),
        ]);
        let outer = AgentBuilder::new(outer_model).tool(inner).build();

        let mut extensions = ToolCallExtensions::new();
        extensions.insert(SessionId("abc-123".to_string()));

        let out = outer
            .prompt("start")
            .tool_extensions(extensions)
            .max_turns(5)
            .await
            .expect("run succeeds");

        assert_eq!(out, "outer done");
        assert_eq!(probe.observed().as_deref(), Some("session:abc-123"));
    }

    #[tokio::test]
    async fn run_context_and_correlation_propagate_through_agent_tool() {
        let observed = Arc::new(Mutex::new(None));
        let inner = AgentBuilder::new(MockCompletionModel::new([
            MockTurn::tool_call("inner-call", "runtime_probe", json!({})),
            MockTurn::text("inner done"),
        ]))
        .name("researcher")
        .tool(RuntimeProbe(observed.clone()))
        .build();
        let outer = AgentBuilder::new(MockCompletionModel::new([
            MockTurn::tool_call("outer-call", "researcher", json!({"prompt": "work"})),
            MockTurn::text("outer done"),
        ]))
        .tool(inner)
        .build();

        let request = outer.prompt("start").max_turns(4);
        let control = request.control_handle();
        assert_eq!(request.await.unwrap(), "outer done");
        let context = observed
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
            .expect("inner tool receives RunContext");
        assert_eq!(context.run_id(), control.run_id());
        assert!(context.current_call_id().is_some_and(|id| !id.is_empty()));
        assert_eq!(context.ancestry(), &["researcher", "runtime_probe"]);
    }
}
