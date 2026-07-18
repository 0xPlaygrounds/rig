#![allow(unused_imports)]

use classic::tool::ToolContext;
use rig_derive::rig_tool;

#[rig_tool]
fn immutable_context(
    #[rig(context)] context: &ToolContext,
) -> Result<(), portable::tool::ToolExecutionError> {
    let _ = context;
    Ok(())
}

fn main() {}
