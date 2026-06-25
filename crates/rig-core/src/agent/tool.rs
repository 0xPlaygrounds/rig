use crate::{
    agent::Agent,
    completion::{CompletionModel, Prompt, PromptError, ToolDefinition},
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

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        let description = format!(
            "
            Prompt a sub-agent to do a task for you.

            Agent name: {name}
            Agent description: {description}
            Agent system prompt: {sysprompt}
            ",
            name = self.name(),
            description = self.description.clone().unwrap_or_default(),
            sysprompt = self.preamble.clone().unwrap_or_default()
        );
        ToolDefinition {
            name: <Self as Tool>::name(self),
            description,
            parameters: json!(schema_for!(AgentToolArgs)),
        }
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
        self.prompt(args.prompt)
            .tool_extensions(extensions.clone())
            .await
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
