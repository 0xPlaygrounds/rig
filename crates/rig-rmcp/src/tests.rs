//! In-process rmcp suites for the handler, the portable adapter (including
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
