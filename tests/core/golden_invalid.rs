//! Matrix G of the effect corpus: invalid tool calls, streamed and
//! ignored. Every cell is scripted from the mock model: no model in the
//! corpus emits a call to a tool that is not in its request (#2449 tried
//! Sonnet 4.6 twice, #2450's dry pass confirmed it), so the shapes here —
//! an unknown call in a stream, an unknown call beside a valid one, a
//! repair, a skip, the runner's `Ignore` policy — are the mock's. The
//! contract is the corpus's all the same: recorded once here, replayed by
//! rig-verify with nothing behind the keys, both interpreters.

use futures::StreamExt;
use rig::agent::{AgentBuilder, MultiTurnStreamItem};
use rig::completion::PromptError;
use rig::effect::EffectFamily;
use rig::message::{AssistantContent, ToolCall, ToolChoice, ToolFunction};
use rig::run::UnhandledInvalidToolCall;
use rig::test_utils::{MockCompletionModel, MockStreamEvent, MockTurn};
use serde_json::json;

use super::golden_recovery::Add;
use crate::goldens::{RepairToAdd, RetryUnknownTool, SKIP_REASON, SkipUnknown, families};

const PREAMBLE: &str = "Use the add tool.";
const PROMPT: &str = "What is 2 + 3?";
const ANSWER: &str = "2 + 3 = 5";

fn args() -> serde_json::Value {
    json!({"x": 2, "y": 3})
}

fn two_calls_turn() -> MockTurn {
    MockTurn::from_contents([
        AssistantContent::ToolCall(ToolCall::from_wire(
            "call-1",
            ToolFunction::new("multiply".to_owned(), args()),
        )),
        AssistantContent::ToolCall(ToolCall::from_wire(
            "call-2",
            ToolFunction::new("add".to_owned(), args()),
        )),
    ])
}

fn stream_turn(events: Vec<MockStreamEvent>) -> Vec<MockStreamEvent> {
    let mut events = events;
    events.push(MockStreamEvent::final_response_with_default_usage());
    events
}

async fn streamed_output(agent: &rig::agent::Agent, max_turns: usize) -> String {
    let mut stream = agent
        .stream_prompt(PROMPT)
        .max_turns(max_turns)
        .stream()
        .await;
    let mut output = None;
    while let Some(item) = stream.next().await {
        if let MultiTurnStreamItem::FinalResponse(response) = item.expect("the stream yields") {
            output = Some(response.output);
        }
    }
    output.expect("a final response")
}

fn tool_result_texts(log: &rig::effect_log::EffectLog, at: usize) -> Vec<String> {
    match &log.records[at].kind {
        rig::effect::EffectKind::Completion { request, .. } => request
            .chat_history
            .iter()
            .filter_map(|message| match message {
                rig::message::Message::User { content } => Some(content.iter()),
                _ => None,
            })
            .flatten()
            .filter_map(|content| match content {
                rig::message::UserContent::ToolResult(result) => Some(
                    result
                        .content
                        .iter()
                        .map(|part| match part {
                            rig::message::ToolResultContent::Text(text) => text.text.clone(),
                            rig::message::ToolResultContent::Json { value } => value.to_string(),
                            other => format!("{other:?}"),
                        })
                        .collect::<String>(),
                ),
                _ => None,
            })
            .collect(),
        other => panic!("a completion, not {other:?}"),
    }
}

// -- retries, streamed ---------------------------------------------------------

/// An unknown call in a streamed turn, retried once with feedback: the
/// retry is a second streamed completion; events kept.
#[tokio::test]
async fn invalid_streamed_retry_once_effect_log_is_the_golden_fixture() {
    let agent = AgentBuilder::new(MockCompletionModel::from_stream_turns([
        stream_turn(vec![MockStreamEvent::tool_call(
            "call-1",
            "multiply",
            args(),
        )]),
        stream_turn(vec![MockStreamEvent::tool_call("call-2", "add", args())]),
        stream_turn(vec![MockStreamEvent::text(ANSWER)]),
    ]))
    .name("golden")
    .preamble(PREAMBLE)
    .tool(Add)
    .add_hook(RetryUnknownTool)
    .record_effects_with_events()
    .build();
    let mut stream = agent
        .stream_prompt(PROMPT)
        .max_turns(4)
        .max_invalid_tool_call_retries(1)
        .stream()
        .await;
    let mut output = None;
    while let Some(item) = stream.next().await {
        if let MultiTurnStreamItem::FinalResponse(response) = item.expect("the stream yields") {
            output = Some(response.output);
        }
    }
    drop(stream);
    assert_eq!(output.as_deref(), Some(ANSWER));
    let log = agent.take_effect_log().expect("recording");
    assert_eq!(
        families(&log),
        [
            EffectFamily::Completion,
            EffectFamily::Completion,
            EffectFamily::Tool,
            EffectFamily::Completion
        ]
    );
    assert!(log.records[0].events.is_some(), "events are kept");
    crate::goldens::golden_effects("mock_invalid_streamed_retry_once", &log);
}

/// Two unknown calls in a row, streamed, retried twice.
#[tokio::test]
async fn invalid_streamed_retry_twice_effect_log_is_the_golden_fixture() {
    let agent = AgentBuilder::new(MockCompletionModel::from_stream_turns([
        stream_turn(vec![MockStreamEvent::tool_call(
            "call-1",
            "multiply",
            args(),
        )]),
        stream_turn(vec![MockStreamEvent::tool_call("call-2", "divide", args())]),
        stream_turn(vec![MockStreamEvent::tool_call("call-3", "add", args())]),
        stream_turn(vec![MockStreamEvent::text(ANSWER)]),
    ]))
    .name("golden")
    .preamble(PREAMBLE)
    .tool(Add)
    .add_hook(RetryUnknownTool)
    .record_effects_with_events()
    .build();
    let mut stream = agent
        .stream_prompt(PROMPT)
        .max_turns(5)
        .max_invalid_tool_call_retries(2)
        .stream()
        .await;
    let mut output = None;
    while let Some(item) = stream.next().await {
        if let MultiTurnStreamItem::FinalResponse(response) = item.expect("the stream yields") {
            output = Some(response.output);
        }
    }
    drop(stream);
    assert_eq!(output.as_deref(), Some(ANSWER));
    let log = agent.take_effect_log().expect("recording");
    assert_eq!(
        families(&log),
        [
            EffectFamily::Completion,
            EffectFamily::Completion,
            EffectFamily::Completion,
            EffectFamily::Tool,
            EffectFamily::Completion
        ]
    );
    crate::goldens::golden_effects("mock_invalid_streamed_retry_twice", &log);
}

// -- the runner's Ignore policy -------------------------------------------------

/// `UnhandledInvalidToolCall::Ignore`, unary: the unknown call is dropped
/// and the turn goes on without it — with nothing else in it, the turn is
/// the run's answer, an empty one. The second scripted turn is never
/// asked for.
#[tokio::test]
async fn invalid_ignore_unary_effect_log_is_the_golden_fixture() {
    let agent = AgentBuilder::new(MockCompletionModel::from_turns([
        MockTurn::tool_call("call-1", "multiply", args()),
        MockTurn::text(ANSWER),
    ]))
    .name("golden")
    .preamble(PREAMBLE)
    .tool(Add)
    .record_effects()
    .build();
    let response = agent
        .prompt(PROMPT)
        .max_turns(3)
        .unhandled_invalid_tool_call(UnhandledInvalidToolCall::Ignore)
        .await
        .expect("the ignored call does not fail the run");
    assert_eq!(
        response.output, "",
        "an ignored-only turn is an empty answer"
    );
    let log = agent.take_effect_log().expect("recording");
    assert_eq!(families(&log), [EffectFamily::Completion]);
    crate::goldens::golden_effects("mock_invalid_ignore_unary", &log);
}

/// The same, streamed with events.
#[tokio::test]
async fn invalid_ignore_streamed_effect_log_is_the_golden_fixture() {
    let agent = AgentBuilder::new(MockCompletionModel::from_stream_turns([
        stream_turn(vec![MockStreamEvent::tool_call(
            "call-1",
            "multiply",
            args(),
        )]),
        stream_turn(vec![MockStreamEvent::text(ANSWER)]),
    ]))
    .name("golden")
    .preamble(PREAMBLE)
    .tool(Add)
    .record_effects_with_events()
    .build();
    let mut stream = agent
        .stream_prompt(PROMPT)
        .max_turns(3)
        .unhandled_invalid_tool_call(UnhandledInvalidToolCall::Ignore)
        .stream()
        .await;
    let mut output = None;
    while let Some(item) = stream.next().await {
        if let MultiTurnStreamItem::FinalResponse(response) = item.expect("the stream yields") {
            output = Some(response.output);
        }
    }
    drop(stream);
    assert_eq!(
        output.as_deref(),
        Some(""),
        "an ignored-only turn is an empty answer"
    );
    let log = agent.take_effect_log().expect("recording");
    assert_eq!(families(&log), [EffectFamily::Completion]);
    crate::goldens::golden_effects("mock_invalid_ignore_streamed", &log);
}

/// An unknown call beside a valid one in one turn under `Ignore`: the
/// valid call runs, the unknown one is dropped.
#[tokio::test]
async fn invalid_mixed_ignore_effect_log_is_the_golden_fixture() {
    let agent = AgentBuilder::new(MockCompletionModel::from_turns([
        two_calls_turn(),
        MockTurn::text(ANSWER),
    ]))
    .name("golden")
    .preamble(PREAMBLE)
    .tool(Add)
    .record_effects()
    .build();
    let response = agent
        .prompt(PROMPT)
        .max_turns(3)
        .unhandled_invalid_tool_call(UnhandledInvalidToolCall::Ignore)
        .await
        .expect("the valid call runs");
    assert_eq!(response.output, ANSWER);
    let log = agent.take_effect_log().expect("recording");
    assert_eq!(
        families(&log),
        [
            EffectFamily::Completion,
            EffectFamily::Tool,
            EffectFamily::Completion
        ]
    );
    crate::goldens::golden_effects("mock_invalid_mixed_ignore", &log);
}

/// The same turn, streamed with events, under `Ignore`.
#[tokio::test]
async fn invalid_mixed_ignore_streamed_effect_log_is_the_golden_fixture() {
    let agent = AgentBuilder::new(MockCompletionModel::from_stream_turns([
        stream_turn(vec![
            MockStreamEvent::tool_call("call-1", "multiply", args()),
            MockStreamEvent::tool_call("call-2", "add", args()),
        ]),
        stream_turn(vec![MockStreamEvent::text(ANSWER)]),
    ]))
    .name("golden")
    .preamble(PREAMBLE)
    .tool(Add)
    .record_effects_with_events()
    .build();
    let mut stream = agent
        .stream_prompt(PROMPT)
        .max_turns(3)
        .unhandled_invalid_tool_call(UnhandledInvalidToolCall::Ignore)
        .stream()
        .await;
    let mut output = None;
    while let Some(item) = stream.next().await {
        if let MultiTurnStreamItem::FinalResponse(response) = item.expect("the stream yields") {
            output = Some(response.output);
        }
    }
    drop(stream);
    assert_eq!(output.as_deref(), Some(ANSWER));
    let log = agent.take_effect_log().expect("recording");
    assert_eq!(
        families(&log),
        [
            EffectFamily::Completion,
            EffectFamily::Tool,
            EffectFamily::Completion
        ]
    );
    crate::goldens::golden_effects("mock_invalid_mixed_ignore_streamed", &log);
}

/// The same turn under `Fail` (the default) with no hook: the run fails
/// at the completion; the valid call does not run.
#[tokio::test]
async fn invalid_mixed_fail_effect_log_is_the_golden_fixture() {
    let agent = AgentBuilder::new(MockCompletionModel::from_turns([two_calls_turn()]))
        .name("golden")
        .preamble(PREAMBLE)
        .tool(Add)
        .record_effects()
        .build();
    let error = agent
        .prompt(PROMPT)
        .max_turns(3)
        .await
        .expect_err("an unknown call fails the run");
    assert!(
        matches!(error, PromptError::UnknownToolCall { .. }),
        "{error:?}"
    );
    let log = agent.take_effect_log().expect("recording");
    assert_eq!(families(&log), [EffectFamily::Completion]);
    crate::goldens::golden_effects("mock_invalid_mixed_fail", &log);
}

// -- repair and skip ------------------------------------------------------------

/// `Repair { tool_name: "add" }`: the unknown call runs as `add` with its
/// arguments.
#[tokio::test]
async fn invalid_repair_to_add_effect_log_is_the_golden_fixture() {
    let agent = AgentBuilder::new(MockCompletionModel::from_turns([
        MockTurn::tool_call("call-1", "multiply", args()),
        MockTurn::text(ANSWER),
    ]))
    .name("golden")
    .preamble(PREAMBLE)
    .tool(Add)
    .add_hook(RepairToAdd)
    .record_effects()
    .build();
    let response = agent
        .prompt(PROMPT)
        .max_turns(3)
        .await
        .expect("the repaired call runs");
    assert_eq!(response.output, ANSWER);
    let log = agent.take_effect_log().expect("recording");
    assert_eq!(
        families(&log),
        [
            EffectFamily::Completion,
            EffectFamily::Tool,
            EffectFamily::Completion
        ]
    );
    match &log.records[1].kind {
        rig::effect::EffectKind::ToolCall { name, .. } => assert_eq!(name, "add"),
        other => panic!("a tool call, not {other:?}"),
    }
    crate::goldens::golden_effects("mock_invalid_repair_to_add", &log);
}

/// `Skip { reason }` under `tool_choice: Auto`: the model sees the reason
/// as the call's result and answers.
#[tokio::test]
async fn invalid_skip_under_auto_effect_log_is_the_golden_fixture() {
    let agent = AgentBuilder::new(MockCompletionModel::from_turns([
        MockTurn::tool_call("call-1", "multiply", args()),
        MockTurn::text(ANSWER),
    ]))
    .name("golden")
    .preamble(PREAMBLE)
    .tool(Add)
    .add_hook(SkipUnknown)
    .record_effects()
    .build();
    let response = agent
        .prompt(PROMPT)
        .max_turns(3)
        .await
        .expect("the skipped call does not fail the run");
    assert_eq!(response.output, ANSWER);
    let log = agent.take_effect_log().expect("recording");
    assert_eq!(
        families(&log),
        [EffectFamily::Completion, EffectFamily::Completion]
    );
    assert_eq!(tool_result_texts(&log, 1), [SKIP_REASON]);
    crate::goldens::golden_effects("mock_invalid_skip_under_auto", &log);
}

/// `Skip` under `tool_choice: None` is refused (`run/mod.rs`): the run
/// fails with `UnknownToolCall`.
#[tokio::test]
async fn invalid_skip_under_none_effect_log_is_the_golden_fixture() {
    let agent = AgentBuilder::new(MockCompletionModel::from_turns([MockTurn::tool_call(
        "call-1",
        "multiply",
        args(),
    )]))
    .name("golden")
    .preamble(PREAMBLE)
    .tool_choice(ToolChoice::None)
    .tool(Add)
    .add_hook(SkipUnknown)
    .record_effects()
    .build();
    let error = agent
        .prompt(PROMPT)
        .max_turns(3)
        .await
        .expect_err("a skip under tool_choice none is refused");
    assert!(
        matches!(error, PromptError::UnknownToolCall { .. }),
        "{error:?}"
    );
    let log = agent.take_effect_log().expect("recording");
    assert_eq!(families(&log), [EffectFamily::Completion]);
    crate::goldens::golden_effects("mock_invalid_skip_under_none", &log);
}

/// A retry under `tool_choice: Required`: the retry is itself a forced
/// call, and the per-run choice forces every later turn too, so the run
/// ends in `MaxTurnsError` after the real call.
#[tokio::test]
async fn invalid_retry_under_required_effect_log_is_the_golden_fixture() {
    let agent = AgentBuilder::new(MockCompletionModel::from_turns([
        MockTurn::tool_call("call-1", "multiply", args()),
        MockTurn::tool_call("call-2", "add", args()),
        MockTurn::tool_call("call-3", "add", args()),
    ]))
    .name("golden")
    .preamble(PREAMBLE)
    .tool_choice(ToolChoice::Required)
    .tool(Add)
    .add_hook(RetryUnknownTool)
    .record_effects()
    .build();
    let error = agent
        .prompt(PROMPT)
        .max_turns(2)
        .max_invalid_tool_call_retries(1)
        .await
        .expect_err("a forced choice never answers in text");
    assert!(
        matches!(error, PromptError::MaxTurnsError { max_turns: 2, .. }),
        "{error:?}"
    );
    let log = agent.take_effect_log().expect("recording");
    assert_eq!(
        families(&log),
        [
            EffectFamily::Completion,
            EffectFamily::Completion,
            EffectFamily::Tool
        ]
    );
    crate::goldens::golden_effects("mock_invalid_retry_under_required", &log);
}

/// A streamed unknown call repaired to `add`.
#[tokio::test]
async fn invalid_streamed_repair_effect_log_is_the_golden_fixture() {
    let agent = AgentBuilder::new(MockCompletionModel::from_stream_turns([
        stream_turn(vec![MockStreamEvent::tool_call(
            "call-1",
            "multiply",
            args(),
        )]),
        stream_turn(vec![MockStreamEvent::text(ANSWER)]),
    ]))
    .name("golden")
    .preamble(PREAMBLE)
    .tool(Add)
    .add_hook(RepairToAdd)
    .record_effects_with_events()
    .build();
    assert_eq!(streamed_output(&agent, 3).await, ANSWER);
    let log = agent.take_effect_log().expect("recording");
    assert_eq!(
        families(&log),
        [
            EffectFamily::Completion,
            EffectFamily::Tool,
            EffectFamily::Completion
        ]
    );
    crate::goldens::golden_effects("mock_invalid_streamed_repair", &log);
}

/// A streamed unknown call skipped.
#[tokio::test]
async fn invalid_streamed_skip_effect_log_is_the_golden_fixture() {
    let agent = AgentBuilder::new(MockCompletionModel::from_stream_turns([
        stream_turn(vec![MockStreamEvent::tool_call(
            "call-1",
            "multiply",
            args(),
        )]),
        stream_turn(vec![MockStreamEvent::text(ANSWER)]),
    ]))
    .name("golden")
    .preamble(PREAMBLE)
    .tool(Add)
    .add_hook(SkipUnknown)
    .record_effects_with_events()
    .build();
    assert_eq!(streamed_output(&agent, 3).await, ANSWER);
    let log = agent.take_effect_log().expect("recording");
    assert_eq!(
        families(&log),
        [EffectFamily::Completion, EffectFamily::Completion]
    );
    assert_eq!(tool_result_texts(&log, 1), [SKIP_REASON]);
    crate::goldens::golden_effects("mock_invalid_streamed_skip", &log);
}
