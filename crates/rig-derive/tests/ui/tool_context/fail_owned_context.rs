use rig_derive::rig_tool;

#[rig_tool]
fn owned_context(
    context: rig_agent::tool::ToolContext,
) -> Result<(), rig_core::tool::ToolExecutionError> {
    let _ = context;
    Ok(())
}

fn main() {}
