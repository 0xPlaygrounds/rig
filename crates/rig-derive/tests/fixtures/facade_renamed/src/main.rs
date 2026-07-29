// Portable tools are imported from the facade `tool` path (`rig::tool`), the
// same path the derive targets for facade-based tools. `Tool` is the facade's
// classic alias for `PortableTool`.
use rig_facade::tool::{PortableTool, Tool, ToolExecutionError};
use rig_macros::rig_tool;

#[rig_tool]
fn portable_echo(value: String) -> Result<String, ToolExecutionError> {
    Ok(value)
}

#[derive(rig_macros::Embed)]
struct Doc {
    #[embed]
    body: String,
}

fn assert_portable<T: PortableTool>() {}
fn assert_classic_alias<T: Tool>() {}
fn assert_embed<T: rig_facade::core::embeddings::Embed>() {}

fn main() {
    assert_portable::<PortableEcho>();
    assert_classic_alias::<PortableEcho>();
    assert_embed::<Doc>();
}
