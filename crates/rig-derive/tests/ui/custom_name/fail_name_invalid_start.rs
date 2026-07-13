#![allow(dead_code)]

use rig_derive::rig_tool;

#[rig_tool(name = "9bad")]
fn invalid_name_start() -> Result<String, std::io::Error> {
    Ok("ok".to_string())
}

fn main() {}
