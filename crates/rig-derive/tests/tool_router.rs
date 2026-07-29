//! Integration tests for `#[derive(ToolRouter)]`.

#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used,
    clippy::unreachable
)]

use rig_agent::agent::run::PendingToolCall;
use rig_core::OneOrMany;
use rig_core::message::{ToolCall, ToolFunction, ToolResultContent, UserContent};
use rig_core::tool::ToolErrorKind;
use rig_derive::{ToolRouter, rig_tool};
use serde_json::json;

#[rig_tool(description = "Add two integers")]
fn add(a: i64, b: i64) -> Result<String, rig_core::tool::ToolExecutionError> {
    Ok(format!("sum:{}", a + b))
}

#[rig_tool(description = "Echo after a delay")]
async fn slow_echo(text: String) -> Result<String, rig_core::tool::ToolExecutionError> {
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    Ok(format!("slow:{text}"))
}

#[rig_tool(description = "Label with a default")]
fn label(name: Option<String>) -> Result<String, rig_core::tool::ToolExecutionError> {
    Ok(format!(
        "label:{}",
        name.unwrap_or_else(|| "default".to_string())
    ))
}

#[derive(ToolRouter)]
struct MyTools {
    slow: SlowEcho,
    add: Add,
    label: Label,
}

fn router() -> MyTools {
    MyTools {
        slow: SlowEcho,
        add: Add,
        label: Label,
    }
}

fn call(id: &str, name: &str, arguments: serde_json::Value) -> ToolCall {
    ToolCall::new(
        id.to_string(),
        ToolFunction::new(name.to_string(), arguments),
    )
}

fn result_text(content: &UserContent) -> (String, String) {
    match content {
        UserContent::ToolResult(result) => {
            let text = match result.content.first_ref() {
                ToolResultContent::Text(text) => text.text.clone(),
                other => panic!("expected text tool-result content, got {other:?}"),
            };
            (result.id.clone(), text)
        }
        other => panic!("expected a tool result, got {other:?}"),
    }
}

#[test]
fn catalog_lists_definitions_in_field_order() {
    let catalog = router().catalog();
    let names: Vec<&str> = catalog
        .definitions
        .iter()
        .map(|def| def.name.as_str())
        .collect();
    assert_eq!(
        names,
        vec!["slow_echo", "add", "label"],
        "definitions must follow field declaration order"
    );
    assert!(
        catalog
            .definitions
            .iter()
            .all(|def| !def.description.is_empty())
    );
    let expected: std::collections::BTreeSet<String> = ["slow_echo", "add", "label"]
        .iter()
        .map(|name| name.to_string())
        .collect();
    assert_eq!(catalog.executable, expected);
}

#[tokio::test]
async fn dispatch_routes_by_name() {
    let result = router()
        .dispatch(&call("1", "add", json!({"a": 2, "b": 3})))
        .await;
    assert!(result.is_success());
    assert_eq!(result.output().render(), "sum:5");
}

#[tokio::test]
async fn dispatch_unknown_name_matches_registry_not_found_shape() {
    let result = router().dispatch(&call("1", "nope", json!({}))).await;
    assert!(result.is_error_kind(ToolErrorKind::NotFound));
    let error = result.error().expect("unknown tool must fail");
    assert_eq!(error.message(), "no tool named `nope` is registered");
    assert_eq!(error.model_feedback(), Some("tool `nope` not found"));
    assert_eq!(result.output().render(), "tool `nope` not found");
}

#[tokio::test]
async fn dispatch_normalizes_null_args_like_the_classic_parser() {
    let result = router()
        .dispatch(&call("1", "label", serde_json::Value::Null))
        .await;
    assert!(result.is_success());
    assert_eq!(result.output().render(), "label:default");
}

#[tokio::test]
async fn dispatch_invalid_args_become_a_model_visible_failure() {
    let result = router()
        .dispatch(&call("1", "add", json!({"a": "not a number"})))
        .await;
    assert!(result.is_error_kind(ToolErrorKind::InvalidArgs));
    assert!(
        result
            .output()
            .as_text()
            .is_some_and(|text| text.starts_with("failed to parse tool arguments:"))
    );
}

#[tokio::test]
async fn dispatch_all_returns_preresolved_results_without_executing() {
    let preresolved = UserContent::tool_result(
        "skipped",
        OneOrMany::one(ToolResultContent::text("preresolved passthrough")),
    );
    // The preresolved call names a real tool with arguments that would fail —
    // proving the tool is never executed.
    let calls = vec![
        PendingToolCall::new(call("skipped", "add", json!("garbage")))
            .with_preresolved_result(preresolved.clone()),
        PendingToolCall::new(call("2", "add", json!({"a": 1, "b": 1}))),
    ];

    let results = router().dispatch_all(&calls, 2).await;
    assert_eq!(results.len(), 2);
    assert_eq!(results[0], preresolved);
    assert_eq!(
        result_text(&results[1]),
        ("2".to_string(), "sum:2".to_string())
    );
}

#[tokio::test]
async fn dispatch_all_preserves_call_order_under_concurrency() {
    // The slow tool finishes last even though it is first in call order;
    // with concurrency 2 the fast calls complete before it.
    let calls = vec![
        PendingToolCall::new(call("first", "slow_echo", json!({"text": "one"}))),
        PendingToolCall::new(call("second", "add", json!({"a": 1, "b": 2}))),
        PendingToolCall::new(call("third", "label", json!({"name": "x"}))),
    ];

    let results = router().dispatch_all(&calls, 2).await;
    let shaped: Vec<(String, String)> = results.iter().map(result_text).collect();
    assert_eq!(
        shaped,
        vec![
            ("first".to_string(), "slow:one".to_string()),
            ("second".to_string(), "sum:3".to_string()),
            ("third".to_string(), "label:x".to_string()),
        ]
    );
}

#[tokio::test]
async fn dispatch_all_unknown_names_become_tool_result_content() {
    let calls = vec![PendingToolCall::new(call("1", "missing", json!({})))];
    let results = router().dispatch_all(&calls, 1).await;
    assert_eq!(
        result_text(&results[0]),
        ("1".to_string(), "tool `missing` not found".to_string())
    );
}
