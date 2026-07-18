#![allow(dead_code)]

use rig_derive::rig_tool;

#[rig_tool(name = "search-docs")]
fn search_docs_impl() -> Result<String, portable::tool::ToolExecutionError> {
    Ok("ok".to_string())
}

fn main() {}
