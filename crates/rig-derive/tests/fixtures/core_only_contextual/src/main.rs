// A contextual tool in a crate that depends only on `rig-core` (under a
// Cargo rename) compiles: `Tool` and `ToolContext` are rig-core's, so no
// runtime crate is needed to author one.
use rig_macros::rig_tool;

#[rig_tool]
fn needs_runtime(
    #[rig(context)] _context: &mut core_runtime::tool::ToolContext,
    value: String,
) -> Result<String, core_runtime::tool::ToolExecutionError> {
    Ok(value)
}

fn main() {
    let tool = NeedsRuntime;
    let _definition = core_runtime::tool::tool_definition(&tool);
}
