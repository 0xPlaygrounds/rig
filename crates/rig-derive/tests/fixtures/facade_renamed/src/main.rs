// Contextual tools are imported from the facade `tool` path (`rig::tool`), the
// same path the derive targets for facade-based contextual tools.
use rig_facade::core::embeddings::Embed;
use rig_facade::tool::{PortableTool, Tool, ToolContext, ToolExecutionError};
use rig_macros::rig_tool;

#[rig_tool]
fn portable_echo(value: String) -> Result<String, ToolExecutionError> {
    Ok(value)
}

#[rig_tool]
fn contextual_echo(
    #[rig(context)] _context: &mut ToolContext,
    value: String,
) -> Result<String, ToolExecutionError> {
    Ok(value)
}

#[derive(rig_macros::Embed)]
struct Doc {
    #[embed]
    body: String,
}

fn assert_portable<T: PortableTool>() {}
fn assert_contextual<T: Tool>() {}
fn assert_embed<T: rig_facade::core::embeddings::Embed>() {}

fn main() {
    assert_portable::<PortableEcho>();
    assert_contextual::<ContextualEcho>();
    assert_embed::<Doc>();
}
