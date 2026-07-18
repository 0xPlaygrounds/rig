#![allow(unused_imports)]

use rig_agent::tool::ToolContext;
use rig_derive::rig_tool;

#[rig_tool]
fn owned_context(
    #[rig(context)]
    context: ToolContext,
) -> Result<(), rig_core::tool::ToolExecutionError> {
    let _ = context;
    Ok(())
}

fn main() {}
