use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::tool::Tool;

/// Arguments for the Think tool
#[derive(Deserialize)]
pub struct ThinkArgs {
    /// The thought to think about
    pub thought: String,
}

/// The Think tool allows agents to stop and think in complex tool use situations.
///
/// This tool provides a dedicated space for structured thinking during complex tasks,
/// particularly when processing external information (e.g., tool call results).
/// It doesn't actually perform any actions or retrieve any information - it just
/// provides a space for the model to reason through complex problems.
///
/// This tool is original derived from the
///  [Think tool](https://anthropic.com/engineering/claude-think-tool) blog post from Anthropic.
#[derive(Deserialize, Serialize)]
pub struct ThinkTool;

impl Tool for ThinkTool {
    const NAME: &'static str = "think";
    type Error = rig::tool::ToolExecutionError;
    type Args = ThinkArgs;
    type Output = String;

    fn description(&self) -> String {
        "Use the tool to think about something. It will not obtain new information
            or change the database, but just append the thought to the log. Use it when complex
            reasoning or some cache memory is needed."
            .to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "A thought to think about."
                }
            },
            "required": ["thought"]
        })
    }

    async fn call(
        &self,
        _context: &mut crate::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, crate::tool::ToolExecutionError> {
        // The think tool doesn't actually do anything except echo back the thought
        // This is intentional - it's just a space for the model to reason through problems
        Ok(args.thought)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::tool_definition;

    #[test]
    fn test_think_tool_definition() {
        let tool = ThinkTool;
        let definition = tool_definition(&tool);

        assert_eq!(definition.name, "think");
        assert!(
            definition
                .description
                .contains("Use the tool to think about something")
        );
    }

    #[tokio::test]
    async fn test_think_tool_call() {
        let tool = ThinkTool;
        let args = ThinkArgs {
            thought: "I need to verify the user's identity before proceeding".to_string(),
        };

        let result = tool
            .call(&mut crate::tool::ToolContext::new(), args)
            .await
            .unwrap();
        assert_eq!(
            result,
            "I need to verify the user's identity before proceeding"
        );
    }
}
