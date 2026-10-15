use std::{
    error::Error,
    fmt::{Display, Formatter},
};

use rig_agent::tool::{Schema, Tool, ToolContext};
use serde::{Deserialize, Serialize};

#[derive(Deserialize, schemars::JsonSchema)]
pub struct OperationArgs {
    /// The first number to add
    x: i32,
    /// The second number to add
    y: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MathError {}

impl Display for MathError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("Math error")
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

    fn description(&self) -> String {
        "Add x and y together".to_string()
    }

    fn parameters(&self) -> Schema {
        Schema::arguments::<Self::Args>()
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        let result = args.x + args.y;
        Ok(result)
    }
}
