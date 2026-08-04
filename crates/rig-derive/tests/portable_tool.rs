#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used
)]
use rig_core::tool::{PortableTool, ToolErrorKind, ToolExecutionError, portable_tool_definition};
use rig_derive::rig_tool;

#[rig_tool(description = "A context-free portable addition tool")]
fn portable_add(left: i32, right: i32) -> Result<i32, ToolExecutionError> {
    Ok(left + right)
}

#[rig_tool(description = "A context-free tool without parameters")]
fn portable_ping() -> Result<String, ToolExecutionError> {
    Ok("pong".to_string())
}

#[rig_tool(description = "A context-free tool that always fails")]
fn portable_fail(message: String) -> Result<String, std::io::Error> {
    Err(std::io::Error::other(message))
}

fn assert_portable<T: PortableTool>() {}

#[tokio::test]
async fn context_free_macro_targets_the_portable_tool_contract() {
    assert_portable::<PortableAdd>();

    let output = PORTABLE_ADD
        .call(PortableAddParameters { left: 2, right: 3 })
        .await;

    assert_eq!(output.ok(), Some(5));
}

#[tokio::test]
async fn portable_constructor_matches_the_trait_definition() {
    let record = PortableAdd.portable();

    assert_eq!(record.name(), "portable_add");
    assert_eq!(record.definition(), portable_tool_definition(&PortableAdd));
}

#[tokio::test]
async fn portable_constructor_round_trips_a_call() {
    let record = PortableAdd.portable();

    let output = record
        .execute(serde_json::json!({ "left": 2, "right": 3 }))
        .await
        .expect("portable call succeeds");

    // A serializable non-string output becomes one structured JSON block.
    assert_eq!(output.as_json(), Some(&serde_json::json!(5)));
}

#[tokio::test]
async fn portable_constructor_treats_null_arguments_as_empty_object() {
    let record = PortablePing.portable();

    let output = record
        .execute(serde_json::Value::Null)
        .await
        .expect("null arguments fall back to an empty object");

    // A string output stays a literal text block.
    assert_eq!(output.as_text(), Some("pong"));
}

#[tokio::test]
async fn portable_constructor_shapes_invalid_arguments() {
    let record = PortableAdd.portable();

    let error = record
        .execute(serde_json::json!({ "left": "two" }))
        .await
        .expect_err("mistyped arguments fail");

    assert_eq!(error.kind(), ToolErrorKind::InvalidArgs);
    assert!(error.message().contains("failed to parse tool arguments"));
}

#[tokio::test]
async fn portable_constructor_normalizes_tool_errors() {
    let record = PortableFail.portable();

    let error = record
        .execute(serde_json::json!({ "message": "boom" }))
        .await
        .expect_err("the tool always fails");

    // The concrete error is normalized like `PortableTool::map_error`'s
    // default: kind-level classification with the source preserved.
    assert_eq!(error.kind(), ToolErrorKind::Other);
    assert!(error.downcast_ref::<std::io::Error>().is_some());
}
