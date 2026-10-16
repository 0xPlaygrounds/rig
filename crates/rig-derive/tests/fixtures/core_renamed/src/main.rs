use core_runtime::tool::{Tool, ToolExecutionError};
use rig_macros::rig_tool;

#[rig_tool]
fn echo(value: String) -> Result<String, ToolExecutionError> {
    Ok(value)
}

#[derive(rig_macros::Embed)]
struct Doc {
    #[embed]
    body: String,
}

fn assert_tool<T: Tool>() {}
fn assert_embed<T: core_runtime::embeddings::Embed>() {}

fn main() {
    assert_tool::<Echo>();
    assert_embed::<Doc>();
}
