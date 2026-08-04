// A parameter typed as the old fully-qualified runtime context path is still
// recognized and rejected with the targeted removal diagnostic.
use rig_derive::rig_tool;

#[rig_tool]
fn wants_runtime_context(
    _context: &mut rig_agent::tool::ToolContext,
    value: String,
) -> Result<String, rig_core::tool::ToolExecutionError> {
    Ok(value)
}

fn main() {}
