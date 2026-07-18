use std::sync::Arc;

use crate::{
    agent::Agent,
    completion::{CompletionModel, Prompt},
    tool::{DynamicTool, ToolExecutionError, ToolOutput},
};
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct AgentToolArgs {
    /// The prompt for the agent to call.
    prompt: String,
}

const DEFAULT_AGENT_TOOL_NAME: &str = "agent_tool";

impl<M: CompletionModel + 'static> Agent<M> {
    /// Convert this agent into a runtime-defined tool.
    ///
    /// The configured agent name becomes the tool name. Unnamed agents use
    /// `agent_tool`. This explicit conversion keeps runtime identity out of the
    /// statically named [`Tool`](crate::tool::Tool) trait.
    pub fn into_tool(self) -> DynamicTool {
        let name = self
            .name
            .clone()
            .unwrap_or_else(|| DEFAULT_AGENT_TOOL_NAME.to_string());
        let description = format!(
            "
            Prompt a sub-agent to do a task for you.

            Agent name: {name}
            Agent description: {description}
            Agent system prompt: {sysprompt}
            ",
            name = name,
            description = self.description.clone().unwrap_or_default(),
            sysprompt = self.preamble.clone().unwrap_or_default()
        );
        let parameters = json!(schema_for!(AgentToolArgs));
        let agent = Arc::new(self);

        DynamicTool::new(name, description, parameters, move |context, args| {
            let agent = Arc::clone(&agent);
            let inherited_context = context.inbound_only();
            Box::pin(async move {
                let args: AgentToolArgs = serde_json::from_value(args).map_err(|error| {
                    ToolExecutionError::invalid_args(format!(
                        "failed to parse agent tool arguments: {error}"
                    ))
                    .with_source(error)
                })?;
                agent
                    .prompt(args.prompt)
                    .tool_context(inherited_context)
                    .await
                    .map(ToolOutput::text)
                    .map_err(ToolExecutionError::from_error)
            })
        })
    }
}

impl<M: CompletionModel + 'static> From<Agent<M>> for DynamicTool {
    fn from(agent: Agent<M>) -> Self {
        agent.into_tool()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::AgentBuilder;
    use crate::test_utils::{MockCompletionModel, MockContextProbeTool, MockTurn, SessionId};
    use crate::tool::ToolContext;

    /// A `ToolContext` set on the outer run propagates into a sub-agent
    /// invoked as a tool, so the inner agent's own tools observe it.
    #[tokio::test]
    async fn context_propagates_into_sub_agent() {
        // Inner agent: calls a context-probing tool, then answers.
        let probe = MockContextProbeTool::default();
        let inner_model = MockCompletionModel::new([
            MockTurn::tool_call("c1", "context_probe", json!({})),
            MockTurn::text("inner done"),
        ]);
        let inner = AgentBuilder::new(inner_model)
            .name("researcher")
            .contextual_tool(probe.clone())
            .build();

        // Outer agent: delegates to the inner agent (registered as the
        // "researcher" tool), then answers.
        let outer_model = MockCompletionModel::new([
            MockTurn::tool_call("c2", "researcher", json!({"prompt": "do research"})),
            MockTurn::text("outer done"),
        ]);
        let outer = AgentBuilder::new(outer_model)
            .dynamic_tool(inner.into_tool())
            .build();

        let mut context = ToolContext::new();
        context.insert(SessionId("abc-123".to_string()));

        let out = outer
            .prompt("start")
            .tool_context(context)
            .max_turns(5)
            .await
            .expect("run succeeds");

        assert_eq!(out, "outer done");
        assert_eq!(probe.observed().as_deref(), Some("session:abc-123"));
    }
}
