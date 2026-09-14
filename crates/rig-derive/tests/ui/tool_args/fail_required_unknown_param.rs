use rig_derive::rig_tool;

// `qurey` does not name a parameter; a typo must not silently advertise a
// nonexistent required field.
#[rig_tool(required(qurey))]
fn search(query: String) -> Result<String, rig_core::tool::ToolExecutionError> {
    Ok(query)
}

fn main() {}
