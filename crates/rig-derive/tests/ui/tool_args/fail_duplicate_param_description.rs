use rig_derive::rig_tool;

// A repeated `params(...)` key used to silently last-win; it must error.
#[rig_tool(params(query = "first", query = "second"))]
fn search(query: String) -> Result<String, rig_core::tool::ToolExecutionError> {
    Ok(query)
}

fn main() {}
