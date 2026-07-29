// A parameter explicitly marked `#[rig(context)]` must fail with the
// targeted removal diagnostic: ToolContext no longer exists.
use rig_derive::rig_tool;

struct LocalContext;

#[rig_tool]
fn wants_runtime_context(
    #[rig(context)] _context: &mut LocalContext,
    value: String,
) -> Result<String, rig_core::tool::ToolExecutionError> {
    Ok(value)
}

fn main() {}
