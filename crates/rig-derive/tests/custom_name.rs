#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used,
    clippy::unreachable
)]

use rig::tool::Tool;
use rig_derive::rig_tool;

#[rig_tool(name = "search-docs")]
fn search_docs_impl() -> Result<String, rig::tool::ToolError> {
    Ok("ok".to_string())
}

#[rig_tool]
fn fallback_name_tool() -> Result<String, rig::tool::ToolError> {
    Ok("fallback".to_string())
}

#[tokio::test]
async fn test_custom_tool_name_overrides_function_name() {
    let tool = SearchDocsImpl;
    let definition = tool.definition(String::default()).await;

    assert_eq!(SearchDocsImpl::NAME, "search-docs");
    assert_eq!(tool.name(), "search-docs");
    assert_eq!(definition.name, "search-docs");
}

#[tokio::test]
async fn test_tool_name_falls_back_to_function_name() {
    let tool = FallbackNameTool;
    let definition = tool.definition(String::default()).await;

    assert_eq!(FallbackNameTool::NAME, "fallback_name_tool");
    assert_eq!(tool.name(), "fallback_name_tool");
    assert_eq!(definition.name, "fallback_name_tool");
}

#[test]
fn test_custom_name_trybuild_cases() {
    let tests = trybuild::TestCases::new();

    tests.pass("tests/ui/custom_name/pass_explicit_name.rs");
    tests.pass("tests/ui/custom_name/pass_fallback_name.rs");
    tests.compile_fail("tests/ui/custom_name/fail_name_non_string.rs");
    tests.compile_fail("tests/ui/custom_name/fail_name_invalid_characters.rs");
    tests.compile_fail("tests/ui/custom_name/fail_name_invalid_start.rs");
    tests.compile_fail("tests/ui/custom_name/fail_name_too_long.rs");
    tests.compile_fail("tests/ui/custom_name/fail_unknown_top_level_argument.rs");
}
