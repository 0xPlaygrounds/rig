#![allow(unused_imports)]

use rig_core::tool::ToolContext;
use rig_derive::rig_tool;

#[rig_tool]
fn immutable_context(context: &ToolContext) -> Result<(), rig_core::tool::ToolExecutionError> {
    let _ = context;
    Ok(())
}

fn main() {}
