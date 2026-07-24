// A contextual tool in a crate that depends only on `rig-core` must fail with
// a targeted diagnostic, not an unresolved `::rig_agent` path.
use rig_macros::rig_tool;

struct LocalContext;

#[rig_tool]
fn needs_runtime(
    #[rig(context)] _context: &mut LocalContext,
    value: String,
) -> Result<String, core_runtime::tool::ToolExecutionError> {
    Ok(value)
}

fn main() {}
