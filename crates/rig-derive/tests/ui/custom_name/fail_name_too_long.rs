#![allow(dead_code)]

use rig_derive::rig_tool;

#[rig_tool(name = "a2345678a2345678a2345678a2345678a2345678a2345678a2345678a2345678x")]
fn invalid_name_length() -> Result<String, rig::tool::ToolError> {
    Ok("ok".to_string())
}

fn main() {}
