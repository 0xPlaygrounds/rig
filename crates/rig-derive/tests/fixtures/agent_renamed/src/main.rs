use agent_runtime::{
    rig_tool,
    tool::{Tool, ToolContext, ToolExecutionError},
};

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

// A fully qualified context path under the renamed dependency is recognized
// without the `#[rig(context)]` marker.
#[rig_tool]
fn contextual_fully_qualified(
    _context: &mut agent_runtime::tool::ToolContext,
    value: String,
) -> Result<String, ToolExecutionError> {
    Ok(value)
}

fn assert_contextual<T: Tool>() {}

fn main() {
    assert_contextual::<ContextualEcho>();
    assert_contextual::<ContextualFullyQualified>();
    assert_contextual::<PortableEcho>();
}
