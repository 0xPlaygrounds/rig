use rig_agent::tool::Tool;
use rig_derive::rig_tool;

#[rig_tool(name = "search-docs")]
fn search_docs_impl() -> Result<String, rig_core::tool::ToolExecutionError> {
    Ok("ok".to_string())
}

#[rig_tool]
fn fallback_name_tool() -> Result<String, rig_core::tool::ToolExecutionError> {
    Ok("fallback".to_string())
}

#[tokio::test]
async fn test_custom_tool_name_overrides_function_name() {
    let tool = SearchDocsImpl;
    let definition = rig_agent::tool::tool_definition(&tool);

    assert_eq!(SearchDocsImpl::NAME, "search-docs");
    assert_eq!(definition.name, "search-docs");
}

#[tokio::test]
async fn test_tool_name_falls_back_to_function_name() {
    let tool = FallbackNameTool;
    let definition = rig_agent::tool::tool_definition(&tool);

    assert_eq!(FallbackNameTool::NAME, "fallback_name_tool");
    assert_eq!(definition.name, "fallback_name_tool");
}
