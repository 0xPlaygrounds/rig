use crate::{
    agent::Agent,
    completion::{CompletionModel, Prompt, PromptError, ToolDefinition},
    tool::Tool,
};
use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
// I did not implement #[non_exhaustive] as there is no initiator method
pub struct AgentToolArgs {
    /// The prompt for the agent to call.
    prompt: String,
}

impl<M: CompletionModel> Tool for Agent<M> {
    const NAME: &'static str = "agent_tool";

    type Error = PromptError;
    type Args = AgentToolArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: <Self as Tool>::name(self),
            description: format!(
                "A tool that allows the agent to call another agent by prompting it. The preamble
                of that agent follows:
                --- 
                {}",
                self.preamble.clone().unwrap_or_default()
            ),
            parameters: serde_json::to_value(schema_for!(AgentToolArgs))
                .expect("converting JSON schema to JSON value should never fail"),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.prompt(args.prompt).await
    }

    fn name(&self) -> String {
        Self::NAME.to_string()
    }
}
