//! Integration tests for `PromptResponse.messages` using mock models.
//! Exercises the real agent loop code path with mocked LLM responses.

use rig::agent::AgentBuilder;
use rig::completion::{Message, Usage};
use rig::message::{AssistantContent, UserContent};
use rig_agent::test_utils::{MockAddTool, MockCompletionModel, MockTurn};

// ---------------------------------------------------------------------------
// Mock model infrastructure
// ---------------------------------------------------------------------------

fn simple_text_turn() -> MockTurn {
    MockTurn::text("hello from mock").with_usage(Usage {
        input_tokens: Some(10),
        output_tokens: Some(5),
        total_tokens: Some(15),
        ..Default::default()
    })
}

fn simple_text_model(turns: usize) -> MockCompletionModel {
    MockCompletionModel::from_turns((0..turns).map(|_| simple_text_turn()))
}

fn tool_then_text_model() -> MockCompletionModel {
    MockCompletionModel::from_turns([
        MockTurn::tool_call("tc_1", "add", serde_json::json!({"x": 2, "y": 3})).with_usage(Usage {
            input_tokens: Some(15),
            output_tokens: Some(8),
            total_tokens: Some(23),
            ..Default::default()
        }),
        MockTurn::text("The answer is 5").with_usage(Usage {
            input_tokens: Some(20),
            output_tokens: Some(4),
            total_tokens: Some(24),
            ..Default::default()
        }),
    ])
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Test 1: `PromptResponse::output` is the accepted assistant text.
#[tokio::test]
async fn standard_prompt_returns_string() {
    let agent = AgentBuilder::new(simple_text_model(1)).build();

    let result: String = agent
        .prompt("hi")
        .await
        .expect("prompt should succeed")
        .output();

    assert_eq!(result, "hello from mock");
}

/// Test 6b: `PromptResponse` implements `Display`, delegating to `output`.
#[tokio::test]
async fn prompt_response_display_shows_output() {
    use rig::agent::PromptResponse;

    let resp = PromptResponse::new("the answer is 42", Usage::default());

    assert_eq!(format!("{resp}"), "the answer is 42");
    // Also works with format args
    assert_eq!(resp.to_string(), "the answer is 42");
}

/// Test 11: `Agent::chat` appends every message produced by a tool roundtrip.
#[tokio::test]
async fn chat_appends_tool_roundtrip_to_history() {
    let agent = AgentBuilder::new(tool_then_text_model())
        .tool(MockAddTool)
        .default_max_turns(2)
        .build();
    let mut history = Vec::<Message>::new();

    let output = agent
        .chat("What is 2 + 3?", &mut history)
        .await
        .expect("chat should succeed")
        .output();

    assert_eq!(output, "The answer is 5");
    assert_eq!(
        history.len(),
        4,
        "expected chat to append [User, Assistant(tool), User(tool result), Assistant], got: {history:#?}"
    );
    assert!(matches!(&history[0], Message::User { .. }));

    match &history[1] {
        Message::Assistant(rig::message::AssistantMessage { content, .. }) => assert!(
            content
                .iter()
                .any(|content| matches!(content, AssistantContent::ToolCall(_))),
            "expected assistant tool call, got: {content:?}"
        ),
        other => panic!("expected Assistant with tool call, got: {other:?}"),
    }

    match &history[2] {
        Message::User { content } => assert!(
            content
                .iter()
                .any(|content| matches!(content, UserContent::ToolResult(_))),
            "expected user tool result, got: {content:?}"
        ),
        other => panic!("expected User with tool result, got: {other:?}"),
    }

    match &history[3] {
        Message::Assistant(rig::message::AssistantMessage { content, .. }) => match content.first()
        {
            Some(AssistantContent::Text(text)) => assert_eq!(text.text, "The answer is 5"),
            other => panic!("expected final assistant text, got: {other:?}"),
        },
        other => panic!("expected final Assistant, got: {other:?}"),
    }
}

/// `memory_append` is absent from a response without memory behind it, and
/// round-trips through serde in both states when set by the driver.
#[test]
fn memory_append_is_absent_without_memory_and_round_trips_through_serde() {
    use rig::agent::{MemoryAppend, PromptResponse};

    let bare = PromptResponse::new("output", Usage::default());
    assert_eq!(bare.memory_append, None);
    let json = serde_json::to_value(&bare).expect("serializes");
    assert!(
        json.get("memory_append").is_none(),
        "no memory, no field: {json}"
    );

    let acknowledged = bare
        .clone()
        .with_memory_append(Some(MemoryAppend::Acknowledged));
    let json = serde_json::to_value(&acknowledged).expect("serializes");
    assert_eq!(
        json["memory_append"],
        serde_json::json!({"status": "acknowledged"})
    );
    let restored: PromptResponse = serde_json::from_value(json).expect("deserializes");
    assert_eq!(restored.memory_append, Some(MemoryAppend::Acknowledged));

    let failed = bare.with_memory_append(Some(MemoryAppend::Failed {
        report: rig::error::ErrorReport::new(rig::error::ErrorKind::MemoryBackend, "boom"),
    }));
    let json = serde_json::to_value(&failed).expect("serializes");
    assert_eq!(json["memory_append"]["status"], "failed");
    let restored: PromptResponse = serde_json::from_value(json).expect("deserializes");
    assert_eq!(restored.memory_append, failed.memory_append);
    assert_eq!(
        restored
            .memory_append()
            .and_then(MemoryAppend::failure)
            .map(|report| report.kind),
        Some(rig::error::ErrorKind::MemoryBackend)
    );
}
