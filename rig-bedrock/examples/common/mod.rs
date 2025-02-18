use std::{
    error::Error,
    fmt::{Display, Formatter},
};

use rig::{completion::ToolDefinition, tool::Tool};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Deserialize)]
pub struct OperationArgs {
    x: i32,
    y: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MathError {}

impl Display for MathError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Math error")
    }
}

impl Error for MathError {}

#[derive(Deserialize, Serialize)]
pub struct Adder;
impl Tool for Adder {
    const NAME: &'static str = "add";

    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "add".to_string(),
            description: "Add x and y together".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "x": {
                        "type": "number",
                        "description": "The first number to add"
                    },
                    "y": {
                        "type": "number",
                        "description": "The second number to add"
                    }
                }
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let result = args.x + args.y;
        Ok(result)
    }
}
