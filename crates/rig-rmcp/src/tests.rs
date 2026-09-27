//! In-process rmcp suites for the handler, the `DynamicTool` adapter (including
//! `_meta` passthrough and result preservation through the per-call
//! `ToolContext`), and the result mapping. rig-agent is a dev-dependency only:
//! its tool server is the reference `ManagedToolSink`/runtime these tests
//! register into.

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

/// A connection that could not be made is a transport failure and a tool
/// list that did not arrive in time is a timeout; both are worth retrying.
#[test]
fn an_mcp_client_error_converts_by_what_failed() {
    use rig_core::error::{ErrorKind, RigError};
    let refused = RigError::from(crate::McpClientError::ConnectionError(
        "connection refused".to_owned(),
    ));
    assert_eq!(refused.kind, ErrorKind::Http);
    assert!(refused.retryable);
    assert_eq!(refused.message, "MCP connection error: connection refused");
    let late = RigError::from(crate::McpClientError::ToolFetchTimeout(
        std::time::Duration::from_secs(5),
    ));
    assert_eq!(late.kind, ErrorKind::Timeout);
    assert!(late.retryable);
    assert_eq!(late.message, "Timed out fetching MCP tool list after 5s");
}
