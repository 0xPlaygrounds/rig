use rig_derive::rig_tool;

// A duplicate `name` used to silently last-win; it must error.
#[rig_tool(name = "first", name = "second")]
fn tool() -> Result<i64, rig_core::tool::ToolExecutionError> {
    Ok(1)
}

fn main() {}
