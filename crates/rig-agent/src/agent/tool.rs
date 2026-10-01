//! Conversion of agents into dynamically named tools.
//!
//! ```
//! use rig_agent::{Agent, core::message::EmptyToolName, tool::DynamicTool};
//! fn delegate(agent: Agent) -> Result<DynamicTool, EmptyToolName> {
//!     agent.into_tool()
//! }
//! ```

use std::sync::Arc;

use crate::{
    agent::Agent,
    tool::{DynamicTool, ToolExecutionError, ToolOutput},
};
use rig_core::message::{EmptyToolName, ToolName};
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
    /// Uses the configured name or `agent_tool`. The tool accepts a JSON `prompt`
    /// string and inherits dispatch context. Invalid arguments and failed runs
    /// become tool execution errors; successful output is returned as text.
    ///
    /// Returns [`EmptyToolName`] when the configured name is empty.
    pub fn into_tool(self) -> Result<DynamicTool, EmptyToolName> {
        let name = ToolName::new(
            self.config
                .name
                .clone()
                .unwrap_or_else(|| DEFAULT_AGENT_TOOL_NAME.to_string()),
        )?;
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

        Ok(DynamicTool::new_with_context(
            name,
            description,
            parameters,
            move |context, args| {
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
                        .map(|response| ToolOutput::text(response.output()))
                        .map_err(ToolExecutionError::from_error)
                })
            },
        ))
    }
}

impl TryFrom<Agent> for DynamicTool {
    type Error = EmptyToolName;

    fn try_from(agent: Agent) -> Result<Self, EmptyToolName> {
        agent.into_tool()
    }
}

#[cfg(test)]
mod tests;
