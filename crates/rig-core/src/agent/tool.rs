use crate::{
    agent::{Agent, RunContext},
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

/// Parent/child delegation depth propagated through tool extensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubagentDepth(pub usize);

/// Default maximum nested agent-as-tool depth.
pub const DEFAULT_SUBAGENT_DEPTH_LIMIT: usize = 8;

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
        let depth = extensions
            .get::<SubagentDepth>()
            .copied()
            .unwrap_or(SubagentDepth(0));
        if depth.0 >= DEFAULT_SUBAGENT_DEPTH_LIMIT {
            return Err(PromptError::prompt_cancelled(
                Vec::new(),
                format!("subagent depth limit ({DEFAULT_SUBAGENT_DEPTH_LIMIT}) reached"),
            ));
        }

        let mut child_extensions = extensions.clone();
        child_extensions.insert(SubagentDepth(depth.0 + 1));
        let runner = self.runner(args.prompt).tool_extensions(child_extensions);
        let child_control = runner.run_control();

        if let Some(parent) = extensions.get::<RunContext>().cloned() {
            let run = runner.run();
            futures::pin_mut!(run);
            let parent_control = parent.control();
            let cancelled = parent_control.cancelled();
            futures::pin_mut!(cancelled);
            match futures::future::select(run, cancelled).await {
                futures::future::Either::Left((result, _)) => {
                    result.map(|response| response.output)
                }
                futures::future::Either::Right(((), run)) => {
                    child_control.cancel();
                    run.await.map(|response| response.output)
                }
            }
        } else {
            runner.run().await.map(|response| response.output)
        }
    }

    fn name(&self) -> String {
        self.name.clone().unwrap_or_else(|| Self::NAME.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::AgentBuilder;
    use crate::test_utils::{MockCompletionModel, MockExtensionsProbeTool, MockTurn, SessionId};

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
}
