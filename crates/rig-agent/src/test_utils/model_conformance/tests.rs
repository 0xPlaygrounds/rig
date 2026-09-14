use super::*;
use crate::{
    completion::Usage,
    test_utils::{MockCompletionModel, MockStreamEvent, MockTurn, mock_final},
};
use rig_core::message::{ToolCall, ToolFunction};

fn tool_call(id: &str, name: &str, arguments: serde_json::Value) -> AssistantContent {
    AssistantContent::ToolCall(ToolCall::from_wire(
        id,
        ToolFunction::new(name.to_string(), arguments),
    ))
}

fn usage(input: u64, output: u64) -> Usage {
    Usage {
        input_tokens: input,
        output_tokens: output,
        total_tokens: input + output,
        ..Usage::new()
    }
}

fn fixture_contract(condition: bool, details: &str) -> Result<(), ScenarioError> {
    if condition {
        Ok(())
    } else {
        Err(ScenarioError::contract("test_fixture", details))
    }
}

#[tokio::test]
async fn parallel_contract_validates_batch_and_correlation() -> Result<(), ScenarioError> {
    let first = MockTurn::from_contents([
        tool_call("call_add", "add", serde_json::json!({"x": 3, "y": 4})),
        tool_call(
            "call_subtract",
            "subtract",
            serde_json::json!({"x": 10, "y": 2}),
        ),
    ]);
    let report = parallel_tools(
        MockCompletionModel::new([first, MockTurn::text("7 and 8")]),
        |builder| builder,
        Some(1),
    )
    .await?;
    fixture_contract(report.tool_calls == 2, "parallel tool-call count")?;
    fixture_contract(report.history_messages >= 4, "parallel history length")?;
    Ok(())
}

#[tokio::test]
async fn zero_argument_and_output_serialization_contracts_pass() -> Result<(), ScenarioError> {
    let zero = zero_argument_tool(
        MockCompletionModel::new([
            MockTurn::tool_call("ping_call", "ping", serde_json::json!({})),
            MockTurn::text(PING_OUTPUT),
        ]),
        |builder| builder,
    )
    .await?;
    fixture_contract(zero.tool_calls == 1, "zero-argument call count")?;

    let first = MockTurn::from_contents([
        tool_call("motto_call", "fetch_motto", serde_json::json!({})),
        tool_call("config_call", "fetch_config", serde_json::json!({})),
    ]);
    let serialized = tool_output_serialization(
        MockCompletionModel::new([first, MockTurn::text("summary")]),
        |builder| builder,
    )
    .await?;
    fixture_contract(serialized.tool_calls == 2, "serialized-output call count")?;
    Ok(())
}

#[tokio::test]
async fn complex_arguments_preserve_nested_unicode_and_escapes() -> Result<(), ScenarioError> {
    let arguments = serde_json::json!({
        "profile": {"name": "Zoë \"Z\"", "tags": ["rust", "東京"]},
        "mode": "careful",
        "note": "line one\nline two",
        "quote": "path C:\\tmp and \"quoted\""
    });
    let report = complex_tool_arguments(
        MockCompletionModel::new([
            MockTurn::tool_call("profile_call", "store_profile", arguments),
            MockTurn::text("stored"),
        ]),
        |builder| builder,
    )
    .await?;
    fixture_contract(report.tool_calls == 1, "complex-argument call count")?;
    Ok(())
}

#[tokio::test]
async fn extraction_contract_requires_fields_and_usage() -> Result<(), ScenarioError> {
    let report = structured_extraction(MockCompletionModel::new([MockTurn::tool_call(
        "submit_call",
        "submit",
        serde_json::json!({
            "first_name": "Ada",
            "last_name": "Lovelace",
            "job": "mathematician"
        }),
    )
    .with_usage(usage(20, 5))]))
    .await?;
    fixture_contract(report.prompt_tokens == 20, "extraction input usage")?;
    fixture_contract(report.generated_tokens == 5, "extraction output usage")?;
    Ok(())
}

#[tokio::test]
async fn streaming_contract_checks_events_history_and_usage() -> Result<(), ScenarioError> {
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("add_call", "add", serde_json::json!({"a": 17, "b": 25})),
            MockStreamEvent::FinalResponse(mock_final(usage(10, 2))),
        ],
        vec![
            MockStreamEvent::text("42"),
            MockStreamEvent::FinalResponse(mock_final(usage(14, 1))),
        ],
    ]);
    let report = streaming_tool(model, |builder| builder).await?;
    fixture_contract(report.prompt_tokens == 24, "streaming input usage")?;
    fixture_contract(report.generated_tokens == 3, "streaming output usage")?;
    Ok(())
}

#[tokio::test]
async fn invalid_recovery_paths_do_not_execute_tools() -> Result<(), ScenarioError> {
    let report = invalid_tool_recovery(
        MockCompletionModel::new([MockTurn::tool_call(
            "invalid-add",
            "add",
            serde_json::json!({ "x": 2, "y": 3 }),
        )]),
        |builder| builder,
    )
    .await?;
    fixture_contract(report.tool_calls == 1, "recovery source call count")?;
    Ok(())
}

#[tokio::test]
async fn hook_rewrites_chain_and_request_patch_is_turn_local() -> Result<(), ScenarioError> {
    let report = hook_rewrites_and_request_patch(
        MockCompletionModel::new([
            MockTurn::tool_call("hook-add", "add", serde_json::json!({ "x": 1, "y": 1 })),
            MockTurn::text("[portable-redacted]"),
        ]),
        |builder| builder,
    )
    .await?;
    fixture_contract(report.tool_calls == 1, "hook execution count")?;
    Ok(())
}

#[tokio::test]
async fn cancellation_and_max_turn_controls_retain_diagnostics() -> Result<(), ScenarioError> {
    let report = cancellation_and_max_turns(
        MockCompletionModel::new([
            MockTurn::tool_call("cancel-add", "add", serde_json::json!({ "x": 20, "y": 22 })),
            MockTurn::tool_call("budget-add", "add", serde_json::json!({ "x": 20, "y": 22 })),
        ]),
        |builder| builder,
    )
    .await?;
    fixture_contract(report.tool_calls == 2, "run-control execution count")?;
    Ok(())
}

#[test]
fn typed_validators_reject_bad_structured_output_and_protocol_leaks() {
    let invalid = decode_structured_output::<ConfigOutput>("invalid_json", "not json");
    assert!(matches!(invalid, Err(ScenarioError::Contract { .. })));

    let messages = vec![Message::Assistant {
        id: None,
        content: vec![AssistantContent::text("visible <tool_call>")],
    }];
    let hygiene = validate_protocol_hygiene(
        "protocol_hygiene",
        "visible <tool_call>",
        &messages,
        &["<tool_call>"],
    );
    assert!(matches!(hygiene, Err(ScenarioError::Contract { .. })));
}

#[test]
fn invalid_tool_diagnostics_require_rejected_call_history() {
    let history = vec![Message::Assistant {
        id: None,
        content: vec![tool_call(
            "bad_call",
            "missing",
            serde_json::json!({"value": 1}),
        )],
    }];
    let error = PromptError::UnknownToolCall {
        tool_name: "missing".to_string(),
        available_tools: vec!["add".to_string()],
        allowed_tools: Vec::new(),
        chat_history: Box::new(history),
    };
    assert!(validate_unknown_tool_failure(&error, "missing", &[]).is_ok());
    assert!(validate_unknown_tool_failure(&error, "other", &[]).is_err());
}
