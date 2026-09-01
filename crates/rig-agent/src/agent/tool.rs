use std::sync::Arc;

use crate::{
    agent::Agent,
    completion::Prompt,
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

impl Agent {
    /// Convert this agent into a runtime-defined tool.
    ///
    /// The configured agent name becomes the tool name. Unnamed agents use
    /// `agent_tool`. This explicit conversion keeps runtime identity out of the
    /// statically named [`Tool`](crate::tool::Tool) trait.
    pub fn into_tool(self) -> DynamicTool {
        let name = self
            .config
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
            description = self.config.description.as_deref().unwrap_or_default(),
            sysprompt = self.config.preamble.as_deref().unwrap_or_default()
        );
        let parameters = json!(schema_for!(AgentToolArgs));
        let agent = Arc::new(self);

        DynamicTool::new(name, description, parameters, move |context, args| {
            let agent = Arc::clone(&agent);
            let inherited_context = context.for_dispatch();
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

impl From<Agent> for DynamicTool {
    fn from(agent: Agent) -> Self {
        agent.into_tool()
    }
}

#[cfg(test)]
mod tests;
