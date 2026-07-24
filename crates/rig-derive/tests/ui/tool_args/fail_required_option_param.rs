use rig_derive::rig_tool;

// schemars drops `Option` fields from `required` and serde deserializes a
// missing `Option` to `None`, so listing one would be silently ignored.
#[rig_tool(required(a, limit))]
fn search(a: i32, limit: Option<i32>) -> Result<i32, rig_core::tool::ToolExecutionError> {
    Ok(a + limit.unwrap_or(0))
}

fn main() {}
