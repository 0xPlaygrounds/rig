use std::sync::Arc;

use crate::{
    agent::Agent,
    completion::{CompletionModel, Prompt},
    tool::{
        IntoToolDyn, ToolCallExtensions, ToolDyn, ToolError, ToolExecutionResult, ToolFailure,
        ToolRuntime,
    },
    wasm_compat::WasmBoxedFuture,
};
use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AgentToolArgs {
    /// The prompt for the agent to call.
    prompt: String,
}

impl<M: CompletionModel + 'static> ToolRuntime for Agent<M> {
    fn call<'a>(&'a self, args: String) -> WasmBoxedFuture<'a, Result<String, ToolError>> {
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
    ) -> WasmBoxedFuture<'a, Result<String, ToolError>> {
        Box::pin(async move {
            let args =
                serde_json::from_str::<AgentToolArgs>(&args).map_err(ToolError::JsonError)?;
            self.prompt(args.prompt)
                .tool_extensions(extensions.clone())
                .await
                .map_err(|err| ToolError::ToolCallError(Box::new(err)))
        })
    }

    fn call_structured<'a>(
        &'a self,
        args: String,
        extensions: &'a ToolCallExtensions,
    ) -> WasmBoxedFuture<'a, ToolExecutionResult> {
        Box::pin(async move {
            match self.call_with_extensions(args, extensions).await {
                Ok(output) => ToolExecutionResult::success(output),
                Err(ToolError::JsonError(err)) => ToolExecutionResult::failed(
                    format!("failed to parse tool arguments: {err}"),
                    ToolFailure::invalid_args(err.to_string()),
                ),
                Err(err) => ToolExecutionResult::failed(
                    err.to_string(),
                    ToolFailure::other(err.to_string()),
                ),
            }
        })
    }
}

impl<M: CompletionModel + 'static> IntoToolDyn for Agent<M> {
    fn into_tool_dyn(self) -> ToolDyn {
        let name = self
            .name
            .clone()
            .unwrap_or_else(|| "agent_tool".to_string());
        let agent = Arc::new(self);
        let description_agent = Arc::clone(&agent);
        let parameters = json!(schema_for!(AgentToolArgs));
        let runtime: Arc<dyn ToolRuntime> = agent;

        ToolDyn::from_runtime(
            name,
            move || {
                format!(
                    "
            Prompt a sub-agent to do a task for you.

            Agent name: {name}
            Agent description: {description}
            Agent system prompt: {sysprompt}
            ",
                    name = description_agent.name(),
                    description = description_agent.description.clone().unwrap_or_default(),
                    sysprompt = description_agent.preamble.clone().unwrap_or_default()
                )
            },
            move || parameters.clone(),
            runtime,
        )
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
