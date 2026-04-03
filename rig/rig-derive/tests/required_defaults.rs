//! Test that `#[rig_tool]` defaults `required` to all parameters when not
//! explicitly specified, matching OpenAI's strict function calling expectations.

use rig::tool::Tool;
use rig_derive::rig_tool;

// No `required(...)` — should default to all params required
#[rig_tool(
    description = "Add two numbers",
    params(a = "First number", b = "Second number")
)]
fn add_implicit(a: i32, b: i32) -> Result<i32, rig::tool::ToolError> {
    Ok(a + b)
}

// Explicit `required(a)` — only `a` should be required
#[rig_tool(
    description = "Add two numbers with optional b",
    params(a = "First number", b = "Second number"),
    required(a)
)]
fn add_explicit(a: i32, b: i32) -> Result<i32, rig::tool::ToolError> {
    Ok(a + b)
}

// No params at all — required should be empty
#[rig_tool(description = "Returns a constant")]
fn constant() -> Result<i32, rig::tool::ToolError> {
    Ok(42)
}

#[tokio::test]
async fn test_required_defaults_to_all_params() {
    let def = AddImplicit.definition(String::default()).await;
    let required = def.parameters["required"].as_array().unwrap();
    let names: Vec<&str> = required.iter().filter_map(|v| v.as_str()).collect();

    assert!(
        names.contains(&"a"),
        "expected 'a' in required, got {names:?}"
    );
    assert!(
        names.contains(&"b"),
        "expected 'b' in required, got {names:?}"
    );
    assert_eq!(names.len(), 2);
}

#[tokio::test]
async fn test_explicit_required_overrides_default() {
    let def = AddExplicit.definition(String::default()).await;
    let required = def.parameters["required"].as_array().unwrap();
    let names: Vec<&str> = required.iter().filter_map(|v| v.as_str()).collect();

    assert_eq!(
        names,
        vec!["a"],
        "expected only 'a' in required, got {names:?}"
    );
}

#[tokio::test]
async fn test_no_params_means_empty_required() {
    let def = Constant.definition(String::default()).await;
    let required = def.parameters["required"].as_array().unwrap();

    assert!(
        required.is_empty(),
        "expected empty required for no-param tool"
    );
}
