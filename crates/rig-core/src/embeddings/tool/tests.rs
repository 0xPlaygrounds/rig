use std::convert::Infallible;

use super::ToolSchema;
use crate::tool::{PortableTool, PortableToolEmbedding, ToolExecutionError};

struct NamedTool;

impl PortableTool for NamedTool {
    const NAME: &'static str = "static_name";

    type Error = rig::tool::ToolExecutionError;

    type Args = ();
    type Output = ();

    fn description(&self) -> String {
        "A statically named tool".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({})
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, ToolExecutionError> {
        Ok(())
    }
}

impl PortableToolEmbedding for NamedTool {
    type InitError = Infallible;
    type Context = ();
    type State = ();

    fn embedding_docs(&self) -> Vec<String> {
        vec!["named tool".to_string()]
    }

    fn context(&self) -> Self::Context {}

    fn init(_state: Self::State, _context: Self::Context) -> Result<Self, Self::InitError> {
        Ok(Self)
    }
}

#[test]
fn try_from_uses_canonical_tool_name() {
    let schema = ToolSchema::try_from(&NamedTool).unwrap();

    assert_eq!(schema.name, NamedTool::NAME);
}
