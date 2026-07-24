use rig_derive::rig_tool;

// `qurey` is a typo for `query`; it must not silently alter the schema.
#[rig_tool(params(qurey = "The search query"))]
fn search(query: String) -> Result<String, rig_core::tool::ToolExecutionError> {
    Ok(query)
}

fn main() {}
