// Contextual tools are imported from the facade `tool` path (`rig::tool`), the
// same path the derive targets for facade-based contextual tools.
use rig_facade::tool::{Tool, ToolContext, ToolExecutionError};
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

// A fully qualified context path under the renamed facade is recognized
// without the `#[rig(context)]` marker.
#[rig_tool]
fn contextual_fully_qualified(
    _context: &mut rig_facade::tool::ToolContext,
    value: String,
) -> Result<String, ToolExecutionError> {
    Ok(value)
}

#[derive(rig_macros::Embed)]
struct Doc {
    #[embed]
    body: String,
}

fn assert_contextual<T: Tool>() {}
fn assert_embed<T: rig_facade::embeddings::Embed>() {}

fn main() {
    assert_contextual::<PortableEcho>();
    assert_contextual::<ContextualEcho>();
    assert_contextual::<ContextualFullyQualified>();
    assert_embed::<Doc>();
}
