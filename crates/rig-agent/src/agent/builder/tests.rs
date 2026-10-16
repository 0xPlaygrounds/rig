use super::*;
use crate::test_utils::{MockAddTool, MockCompletionModel, MockSubtractTool};

/// Tools added through the builder join a shared registry at build, after
/// the ones it already holds.
#[test]
fn builder_tools_join_a_shared_registry() {
    let shared = ToolServer::new().tool(MockSubtractTool).run();
    let agent = AgentBuilder::new(MockCompletionModel::text("ok"))
        .tool(MockAddTool)
        .tool_server_handle(shared.clone())
        .build();
    assert_eq!(
        shared.snapshot().names().collect::<Vec<_>>(),
        ["subtract", "add"]
    );
    assert_eq!(agent.tool_server_handle().snapshot().names().count(), 2);
}
