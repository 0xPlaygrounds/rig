//! Test that `#[rig_tool]` defaults `required` to all parameters when not
//! explicitly specified, matching OpenAI's strict function calling expectations.

#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used,
    clippy::unreachable
)]

use rig_derive::rig_tool;

// No `required(...)` — should default to all params required
#[rig_tool(
    description = "Add two numbers",
    params(a = "First number", b = "Second number")
)]
fn add_implicit(a: i32, b: i32) -> Result<i32, rig_core::tool::ToolExecutionError> {
    Ok(a + b)
}

// Explicit `required(a)` — only `a` should be required
#[rig_tool(
    description = "Add two numbers with optional b",
    params(a = "First number", b = "Second number"),
    required(a)
)]
fn add_explicit(a: i32, b: i32) -> Result<i32, rig_core::tool::ToolExecutionError> {
    Ok(a + b)
}

// Explicit `required()` means all params are optional.
#[rig_tool(description = "Search optionally", required())]
fn search_optional(limit: Option<i32>) -> Result<String, rig_core::tool::ToolExecutionError> {
    Ok(format!("{limit:?}"))
}

// No params at all — required should be empty
#[rig_tool(description = "Returns a constant")]
fn constant() -> Result<i32, rig_core::tool::ToolExecutionError> {
    Ok(42)
}

#[tokio::test]
async fn test_required_defaults_to_all_params() {
    let def = rig_agent::tool::tool_definition(&AddImplicit);
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
    let def = rig_agent::tool::tool_definition(&AddExplicit);
    let required = def.parameters["required"].as_array().unwrap();
    let names: Vec<&str> = required.iter().filter_map(|v| v.as_str()).collect();

    assert_eq!(
        names,
        vec!["a"],
        "expected only 'a' in required, got {names:?}"
    );
}

#[tokio::test]
async fn test_explicit_empty_required_overrides_default() {
    let def = rig_agent::tool::tool_definition(&SearchOptional);
    let required = def.parameters["required"].as_array().unwrap();

    assert!(
        required.is_empty(),
        "expected explicit required() to make all params optional, got {required:?}"
    );
}

/// The schema and the deserializer must agree: a parameter omitted from an
/// explicit `required(...)` list deserializes via its `Default` when the model
/// leaves it out, instead of failing at runtime.
#[tokio::test]
async fn test_param_omitted_from_required_deserializes_via_default() {
    let params: AddExplicitParameters =
        serde_json::from_value(serde_json::json!({"a": 7})).unwrap();
    assert_eq!(params.a, 7);
    assert_eq!(params.b, 0, "omitted non-required param should use Default");
}

/// A parameter listed in an explicit `required(...)` list stays required for
/// the deserializer as well.
#[tokio::test]
async fn test_param_listed_in_required_stays_required_for_deserialization() {
    let error = serde_json::from_value::<AddExplicitParameters>(serde_json::json!({"b": 7}));
    assert!(
        error.is_err(),
        "omitting a required param should fail deserialization"
    );
}

#[tokio::test]
async fn test_no_params_means_empty_required() {
    let def = rig_agent::tool::tool_definition(&Constant);
    let required = def.parameters["required"].as_array().unwrap();

    assert!(
        required.is_empty(),
        "expected empty required for no-param tool"
    );
}
