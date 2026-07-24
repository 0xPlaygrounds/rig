use agent_runtime::{
    core::Embed,
    rig_tool,
    tool::{Tool, ToolContext, ToolExecutionError},
};

#[derive(Embed)]
struct EmbeddedDocument {
    #[embed]
    body: String,
}

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

// A fully qualified context path under the renamed dependency is recognized
// without the `#[rig(context)]` marker.
#[rig_tool]
fn contextual_fully_qualified(
    _context: &mut agent_runtime::tool::ToolContext,
    value: String,
) -> Result<String, ToolExecutionError> {
    Ok(value)
}

fn assert_contextual<T: Tool>() {}
fn assert_portable<T: agent_runtime::core::tool::PortableTool>() {}
fn assert_embed<T: agent_runtime::core::embeddings::Embed>() {}

fn main() {
    assert_contextual::<ContextualEcho>();
    assert_contextual::<ContextualFullyQualified>();
    assert_portable::<PortableEcho>();
    assert_embed::<EmbeddedDocument>();

    // Portable core items stay reachable through the explicit `core` namespace
    // even under a renamed `rig-agent` dependency.
    let _reachable: Option<agent_runtime::core::OneOrMany<u8>> = None;
}
