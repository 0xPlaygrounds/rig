use serde::{Deserialize, Serialize};

use crate::{
    completion::{CompletionModel, Prompt, PromptError, ToolDefinition},
    tool::Tool,
};

use super::Agent;

/// Trait for tools that can be used by the agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentToolArgs {
    prompt: String,
}

impl<M: CompletionModel> Tool for Agent<M> {
    const NAME: &'static str = "agent_tool";

    type Error = PromptError;
    type Args = AgentToolArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: self.name(),
            description: format!(
                "A tool that allows the agent to call another agent by prompting it. The preamble
                of that agent follows:
                --- 
                {}",
                self.preamble.clone()
            ),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The prompt for the agent to call."
                    }
                },
                "required": ["prompt"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        futures::executor::block_on(self.prompt(args.prompt).send())
    }

    fn name(&self) -> String {
        self.name.clone().unwrap_or_else(|| Self::NAME.to_string())
    }
}
