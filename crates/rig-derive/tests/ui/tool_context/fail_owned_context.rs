use rig_derive::rig_tool;

#[rig_tool]
fn owned_context(
    context: classic::tool::ToolContext,
) -> Result<(), portable::tool::ToolExecutionError> {
    let _ = context;
    Ok(())
}

fn main() {}
