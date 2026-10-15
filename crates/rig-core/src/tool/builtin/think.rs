use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::tool::{PortableTool, Schema};

/// Arguments for the Think tool
#[derive(Deserialize, JsonSchema)]
pub struct ThinkArgs {
    /// A thought to think about.
    pub thought: String,
}

/// Error type for the Think tool.
#[derive(Debug, thiserror::Error)]
#[error("Think tool error: {0}")]
pub struct ThinkError(String);

/// Returns the supplied thought unchanged, without I/O or state mutation.
#[derive(Deserialize, Serialize)]
pub struct ThinkTool;

impl PortableTool for ThinkTool {
    const NAME: &'static str = "think";
    type Error = ThinkError;
    type Args = ThinkArgs;
    type Output = String;

    fn description(&self) -> String {
        "Use the tool to think about something. It will not obtain new information
            or change the database, but just append the thought to the log. Use it when complex
            reasoning or some cache memory is needed."
            .to_string()
    }

    fn parameters(&self) -> Schema {
        Schema::arguments::<ThinkArgs>()
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(args.thought)
    }
}

#[cfg(test)]
mod tests;
