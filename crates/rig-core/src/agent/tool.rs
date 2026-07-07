use crate::{
    agent::Agent,
    completion::{CompletionModel, Prompt},
    tool::{IntoToolDyn, ToolCallExtensions, ToolDyn, ToolError, ToolRuntime},
};
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AgentToolArgs {
    /// The prompt for the agent to call.
    prompt: String,
}

impl<M: CompletionModel + 'static> Agent<M> {
    fn agent_tool_name(&self) -> String {
        self.name
            .clone()
            .unwrap_or_else(|| "agent_tool".to_string())
    }

    fn agent_tool_description(&self) -> String {
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

    fn agent_tool_parameters(&self) -> serde_json::Value {
        json!(schema_for!(AgentToolArgs))
    }
}

impl<M: CompletionModel + 'static> ToolRuntime for Agent<M> {
    fn description(&self) -> String {
        self.agent_tool_description()
    }

    fn parameters(&self) -> serde_json::Value {
        self.agent_tool_parameters()
    }

    fn call<'a>(
        &'a self,
        args: String,
    ) -> crate::wasm_compat::WasmBoxedFuture<'a, Result<String, ToolError>> {
        self.call_with_extensions(args, &ToolCallExtensions::EMPTY)
    }

    /// Propagate the caller's [`ToolCallExtensions`] into the sub-agent run, so the
    /// inner agent's own tools observe them too (sub-agent delegation / A2A
    /// chains). Without this, a sub-agent invoked as a tool would start with
    /// empty extensions.
    fn call_with_extensions<'a>(
        &'a self,
        args: String,
        extensions: &'a ToolCallExtensions,
    ) -> crate::wasm_compat::WasmBoxedFuture<'a, Result<String, ToolError>> {
        Box::pin(async move {
            let args =
                serde_json::from_str::<AgentToolArgs>(&args).map_err(ToolError::JsonError)?;
            self.prompt(args.prompt)
                .tool_extensions(extensions.clone())
                .await
                .map_err(|err| ToolError::ToolCallError(Box::new(err)))
        })
    }
}

impl<M: CompletionModel + 'static> IntoToolDyn for Agent<M> {
    fn into_tool_dyn(self) -> ToolDyn {
        let name = self.agent_tool_name();
        ToolDyn::from_runtime(name, self)
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

        // Outer agent invokes the inner agent as the named "researcher" tool), then answers.
        let outer_model = MockCompletionModel::new([
            MockTurn::tool_call("c2", "researcher", json!({"prompt": "do research"})),
            MockTurn::text("outer done"),
        ]);
        let outer = AgentBuilder::new(outer_model).tool(inner).build();

        let mut extensions = ToolCallExtensions::new();
        extensions.insert(SessionId("abc".to_string()));
        let response = outer
            .prompt("start")
            .tool_extensions(extensions)
            .await
            .expect("run succeeds");

        assert_eq!(response, "outer done");
        assert_eq!(probe.observations(), vec!["session:abc".to_string()]);
    }
}
