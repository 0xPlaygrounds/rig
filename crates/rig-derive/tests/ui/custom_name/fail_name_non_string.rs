#![allow(dead_code)]

use rig_derive::rig_tool;

#[rig_tool(name = 123)]
fn invalid_name_value() -> Result<String, rig_core::tool::ToolError> {
    Ok("ok".to_string())
}

fn main() {}
