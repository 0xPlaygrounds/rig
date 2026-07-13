use rig_core::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Deserialize)]
pub struct OperationArgs {
    x: i32,
    y: i32,
}

#[derive(Deserialize, Serialize)]
pub struct Adder;
impl Tool for Adder {
    const NAME: &'static str = "add";
    type Args = OperationArgs;
    type Output = i32;

    fn description(&self) -> String {
        "Add x and y together".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
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
        })
    }

    async fn call(
        &self,
        _context: &mut rig_core::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, rig_core::tool::ToolExecutionError> {
        let result = args.x + args.y;
        Ok(result)
    }
}
