#![allow(unused_imports)]

use rig_core::tool::ToolContext;
use rig_derive::rig_tool;

#[rig_tool]
fn multiple_contexts(
    first: &mut ToolContext,
    second: &mut ToolContext,
) -> Result<(), rig_core::tool::ToolExecutionError> {
    let _ = (first, second);
    Ok(())
}

fn main() {}
