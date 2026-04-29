#![allow(dead_code)]

use rig_derive::rig_tool;

#[rig_tool(name = "bad name!")]
fn invalid_name_characters() -> Result<String, rig::tool::ToolError> {
    Ok("ok".to_string())
}

fn main() {}
