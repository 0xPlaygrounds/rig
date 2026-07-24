// Contextual tools are imported from the facade `tool` path (`rig::tool`), the
// same path the derive targets for facade-based contextual tools.
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

// Fully qualified context paths under the renamed facade are recognized
// without the `#[rig(context)]` marker — both the facade `tool` path and the
// explicit `agent::tool` path.
#[rig_tool]
fn contextual_fully_qualified(
    _context: &mut rig_facade::tool::ToolContext,
    value: String,
) -> Result<String, ToolExecutionError> {
    Ok(value)
}

#[rig_tool]
fn contextual_agent_path(
    _context: &mut rig_facade::agent::tool::ToolContext,
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
    assert_contextual::<ContextualFullyQualified>();
    assert_contextual::<ContextualAgentPath>();
    assert_embed::<Doc>();
}
