#![allow(dead_code)]

use rig_derive::rig_tool;

#[rig_tool]
fn fallback_name_tool() -> Result<String, std::io::Error> {
    Ok("fallback".to_string())
}

fn main() {}
