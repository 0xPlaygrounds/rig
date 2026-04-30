#![allow(dead_code)]

use rig_derive::rig_tool;

#[rig_tool(name = "9bad")]
fn invalid_name_start() -> Result<String, rig_core::tool::ToolError> {
    Ok("ok".to_string())
}

fn main() {}
