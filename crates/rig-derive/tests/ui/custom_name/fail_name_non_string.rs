#![allow(dead_code)]

use rig_derive::rig_tool;

#[rig_tool(name = 123)]
fn invalid_name_value() -> Result<String, std::io::Error> {
    Ok("ok".to_string())
}

fn main() {}
