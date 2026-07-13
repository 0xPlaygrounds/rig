#![allow(dead_code)]

use rig_derive::rig_tool;

#[rig_tool(nam = "search-docs")]
fn unknown_argument() -> Result<String, std::io::Error> {
    Ok("ok".to_string())
}

fn main() {}
