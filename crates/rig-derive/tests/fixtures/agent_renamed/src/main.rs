use agent_runtime::{core::Embed, rig_tool, tool::ToolExecutionError};

#[derive(Embed)]
struct EmbeddedDocument {
    #[embed]
    body: String,
}

#[rig_tool]
fn portable_echo(value: String) -> Result<String, ToolExecutionError> {
    Ok(value)
}

fn assert_portable<T: agent_runtime::tool::PortableTool>() {}
fn assert_embed<T: agent_runtime::core::embeddings::Embed>() {}

fn main() {
    assert_portable::<PortableEcho>();
    assert_embed::<EmbeddedDocument>();

    // Portable core items stay reachable through the explicit `core` namespace
    // even under a renamed `rig-agent` dependency.
    let _reachable: Option<agent_runtime::core::OneOrMany<u8>> = None;
}
