use crate::{
    test_utils::{MockAddTool, MockSubtractTool, MockToolIndex},
    tool::{
        ToolContext, ToolExecutionError, ToolSet,
        server::{ToolServer, ToolServerHandle},
    },
};

async fn execute_tool(
    handle: &ToolServerHandle,
    name: &str,
    args: &str,
) -> Result<String, ToolExecutionError> {
    execute_tool_with_context(handle, name, args, &mut ToolContext::new()).await
}

/// The sync snapshot and the async, prompt-less `tool_defs` read the
/// same always-exposed registry in the same order.
#[tokio::test]
async fn sync_snapshot_matches_async_prompt_less_read() {
    let handle = ToolServer::new()
        .tool(MockAddTool)
        .tool(MockSubtractTool)
        .run();

    let sync_defs = handle.static_tool_defs();
    let async_defs = handle.tool_defs(None).await.unwrap();
    assert_eq!(sync_defs.len(), 2);
    assert_eq!(sync_defs, async_defs);

    let snapshot = handle.snapshot();
    assert_eq!(snapshot.definitions(), sync_defs.as_slice());
    assert_eq!(
        snapshot.names().collect::<Vec<_>>(),
        vec!["add", "subtract"]
    );
    assert_eq!(snapshot.len(), 2);
    assert!(!snapshot.is_empty());
}

async fn execute_tool_with_context(
    handle: &ToolServerHandle,
    name: &str,
    args: &str,
    context: &mut ToolContext,
) -> Result<String, ToolExecutionError> {
    let result = handle.execute(name, args, context).await;
    match result.error() {
        Some(error) => Err(error.clone()),
        None => Ok(result.output().render()),
    }
}

#[tokio::test]
pub async fn duplicate_registration_advertises_one_definition() {
    let handle = ToolServer::new().tool(MockAddTool).run();
    handle.add_tool(MockAddTool);

    let mut toolset = ToolSet::default();
    toolset.add_tool(MockAddTool);
    handle.add_tools(toolset);

    let defs = handle.tool_defs(None).await.unwrap();
    assert_eq!(
        defs.len(),
        1,
        "re-registering a name must not advertise duplicate declarations"
    );
    assert_eq!(defs[0].name, "add");
}

#[tokio::test]
pub async fn test_toolserver_retrieved_tools_missing_implementation() {
    // Create a mock index that returns a tool ID that doesn't exist in the toolset
    let mock_index = MockToolIndex::new(["nonexistent_tool"]);

    // Build server with only static tool, but dynamic index references missing tool
    let server =
        ToolServer::new()
            .tool(MockAddTool)
            .retrieved_tools(1, mock_index, ToolSet::default());

    let handle = server.run();

    // Test with Some prompt - should only return static tool since dynamic tool is missing
    let res = handle
        .tool_defs(Some("some query".to_string()))
        .await
        .unwrap();
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].name, "add");
}

#[tokio::test]
async fn execute_classifies_a_missing_tool_as_not_found() {
    let handle = ToolServer::new().tool(MockAddTool).run();
    let error = execute_tool(&handle, "does_not_exist", "{}")
        .await
        .unwrap_err();
    assert_eq!(error.kind(), crate::tool::ToolErrorKind::NotFound);
    assert!(
        error
            .model_feedback()
            .is_some_and(|feedback| feedback.contains("does_not_exist"))
    );
}
