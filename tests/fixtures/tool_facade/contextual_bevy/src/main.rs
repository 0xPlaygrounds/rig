use std::convert::Infallible;

use rig::agent::tool::{Tool, ToolContext};
use serde::Deserialize;

#[derive(Deserialize)]
struct Arguments;

struct ContextualClassicTool;

impl Tool for ContextualClassicTool {
    const NAME: &'static str = "contextual_classic_tool";
    type Args = Arguments;
    type Output = ();
    type Error = Infallible;

    fn description(&self) -> String {
        "classic-only contextual tool".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({"type": "object"})
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        _arguments: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Ok(())
    }
}

fn must_not_compile(
    runtime: &mut rig::bevy::LocalRuntime,
    agent: rig::bevy::AgentId,
) {
    let _ = runtime.install_tool(agent, ContextualClassicTool);
}

fn main() {}
