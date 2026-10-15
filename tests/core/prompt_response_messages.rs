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
    MockTurn::text("hello from mock").with_usage(
        Usage::new()
            .input_tokens(10)
            .output_tokens(5)
            .total_tokens(15),
    )
}

fn simple_text_model(turns: usize) -> MockCompletionModel {
    MockCompletionModel::from_turns((0..turns).map(|_| simple_text_turn()))
}

fn tool_then_text_model() -> MockCompletionModel {
    MockCompletionModel::from_turns([
        MockTurn::tool_call("tc_1", "add", serde_json::json!({"x": 2, "y": 3})).with_usage(
            Usage::new()
                .input_tokens(15)
                .output_tokens(8)
                .total_tokens(23),
        ),
        MockTurn::text("The answer is 5").with_usage(
            Usage::new()
                .input_tokens(20)
                .output_tokens(4)
                .total_tokens(24),
        ),
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
