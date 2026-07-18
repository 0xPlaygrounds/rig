#![allow(unused_imports)]

use classic::tool::ToolContext;
use rig_derive::rig_tool;

#[rig_tool(params(context = "host context"))]
fn context_in_params(
    #[rig(context)]
    context: &mut ToolContext,
    query: String,
) -> Result<String, portable::tool::ToolExecutionError> {
    let _ = context;
    Ok(query)
}

fn main() {}
