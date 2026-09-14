#![allow(unused_imports)]

use rig_agent::tool::ToolContext;
use rig_derive::rig_tool;

// The runtime context parameter must be a nameable binding; a wildcard `_`
// is rejected — name it `_context` instead.
#[rig_tool]
fn wildcard_context(
    #[rig(context)] _: &mut ToolContext,
    value: String,
) -> Result<String, rig_core::tool::ToolExecutionError> {
    Ok(value)
}

fn main() {}
