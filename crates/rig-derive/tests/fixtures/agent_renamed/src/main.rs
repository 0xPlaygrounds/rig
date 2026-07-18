use agent_runtime::tool::{Tool, ToolContext, ToolExecutionError};
use rig_macros::rig_tool;

#[rig_tool]
fn contextual_echo(
    #[rig(context)] _context: &mut ToolContext,
    value: String,
) -> Result<String, ToolExecutionError> {
    Ok(value)
}

fn assert_contextual<T: Tool>() {}

fn main() {
    assert_contextual::<ContextualEcho>();
}
