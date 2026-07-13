#![allow(unused_imports)]

use rig_core::tool::ToolContext;
use rig_derive::rig_tool;

#[rig_tool(params(context = "host context"))]
fn context_in_params(
    #[rig(context)]
    context: &mut ToolContext,
    query: String,
) -> Result<String, rig_core::tool::ToolExecutionError> {
    let _ = context;
    Ok(query)
}

fn main() {}
