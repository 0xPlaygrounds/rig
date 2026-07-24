#![allow(unused_imports)]

use rig_agent::tool::ToolContext;
use rig_derive::rig_tool;

#[rig_tool]
fn multiple_contexts(
    #[rig(context)]
    first: &mut ToolContext,
    #[rig(context)]
    second: &mut ToolContext,
) -> Result<(), rig_core::tool::ToolExecutionError> {
    let _ = (first, second);
    Ok(())
}

fn main() {}
