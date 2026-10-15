//! In-process rmcp suites for the handler, the `DynamicTool` adapter (including
//! `_meta` passthrough and result preservation through the per-call
//! `ToolContext`), and the result mapping. rig-agent is a dev-dependency only:
//! its tool server is the reference `ManagedToolSink`/runtime these tests
//! register into.

#[cfg(test)]
mod connection;

#[cfg(test)]
mod dispatch;

#[cfg(test)]
mod migrated_tests;

// Compile-time thread-safety contract: rmcp's `ClientHandler` requires it, and
// rig-agent's `ToolServerHandle` is the sink the docs recommend.
const _: fn() = || {
    fn assert_send_sync_static<T: Send + Sync + 'static>() {}
    assert_send_sync_static::<crate::McpClientHandler<rig_agent::tool::server::ToolServerHandle>>();
};

/// An MCP tool's hints become rig's tool annotations, field for field, and
/// a tool without hints states nothing.
#[test]
fn mcp_hints_become_tool_annotations() {
    use std::sync::Arc;

    use rmcp::model::{Tool, ToolAnnotations};

    let mut tool = Tool::new("read", "Read a file", Arc::new(serde_json::Map::new()));
    assert!(crate::native::annotations(&tool).is_empty());
    let mut hints = ToolAnnotations::new();
    hints.read_only_hint = Some(true);
    hints.open_world_hint = Some(false);
    hints.title = Some("Read".to_owned());
    tool.annotations = Some(hints);
    let annotations = crate::native::annotations(&tool);
    assert!(annotations.read_only() && !annotations.open_world() && !annotations.destructive());
    assert_eq!(annotations.title.as_deref(), Some("Read"));
}
