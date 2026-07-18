use rig_facade::tool::{PortableTool, Tool, ToolContext, ToolExecutionError};
use rig_macros::rig_tool;

#[rig_tool]
fn portable_echo(value: String) -> Result<String, ToolExecutionError> {
    Ok(value)
}

#[rig_tool]
fn contextual_echo(
    #[rig(context)] _context: &mut ToolContext,
    value: String,
) -> Result<String, ToolExecutionError> {
    Ok(value)
}

fn assert_portable<T: PortableTool>() {}
fn assert_contextual<T: Tool>() {}

fn main() {
    assert_portable::<PortableEcho>();
    assert_contextual::<ContextualEcho>();
}
