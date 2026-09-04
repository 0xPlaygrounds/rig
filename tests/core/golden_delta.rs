//! Matrix K's mock cells: the delta wire. A tool call that arrives as a
//! name delta and argument deltas (the shape of the openai chat-completions
//! and gemini interactions wires) is not the same turn as a completed
//! block: the assembler validates the name when it arrives, buffers
//! arguments that arrive before it, and an invalid call surfaces at the
//! name delta with the arguments seen so far. Every resolution Matrix G
//! recorded on the block shape is recorded here on the delta shape,
//! scripted from the mock model (no model in the corpus emits an
//! unadvertised call on request).

use futures::StreamExt;
use rig::agent::{AgentBuilder, MultiTurnStreamItem};
use rig::completion::PromptError;
use rig::effect::EffectFamily;
use rig::error::ErrorKind;
use rig::run::{OutputMode, UnhandledInvalidToolCall};
use rig::test_utils::{MockCompletionModel, MockStreamEvent};
use serde_json::json;

use super::golden_recovery::Add;
use crate::goldens::{
    RepairToAdd, RetryUnknownTool, STOP_ON_TOOL_ARGUMENTS_DELTA, STOP_ON_TOOL_NAME_DELTA,
    SkipUnknown, StopOnToolArgumentsDelta, StopOnToolNameDelta, event_schema, families,
};

const PREAMBLE: &str = "Use the add tool.";
const PROMPT: &str = "What is 2 + 3?";
const ANSWER: &str = "2 + 3 = 5";

fn stream_turn(events: Vec<MockStreamEvent>) -> Vec<MockStreamEvent> {
    let mut events = events;
    events.push(MockStreamEvent::final_response_with_default_usage());
    events
}

/// A call streamed as its name, then its arguments in two fragments.
fn delta_call(id: &str, name: &str) -> Vec<MockStreamEvent> {
    vec![
        MockStreamEvent::tool_call_name_delta(id, name),
        MockStreamEvent::tool_call_arguments_delta(id, "{\"x\": 2, "),
        MockStreamEvent::tool_call_arguments_delta(id, "\"y\": 3}"),
        MockStreamEvent::tool_call_end(id),
    ]
}

/// The same call with its arguments streamed before its name.
fn delta_call_arguments_first(id: &str, name: &str) -> Vec<MockStreamEvent> {
    vec![
        MockStreamEvent::tool_call_arguments_delta(id, "{\"x\": 2, "),
        MockStreamEvent::tool_call_arguments_delta(id, "\"y\": 3}"),
        MockStreamEvent::tool_call_name_delta(id, name),
        MockStreamEvent::tool_call_end(id),
    ]
}

async fn streamed(
    agent: &rig::agent::Agent,
    max_turns: usize,
    unhandled: UnhandledInvalidToolCall,
    retries: usize,
) -> Result<String, rig::agent::StreamingError> {
    let mut stream = agent
        .stream_prompt(PROMPT)
        .max_turns(max_turns)
        .max_invalid_tool_call_retries(retries)
        .unhandled_invalid_tool_call(unhandled)
        .stream()
        .await;
    let mut output = None;
    let mut failure = None;
    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::FinalResponse(response)) => output = Some(response.output),
            Ok(_) => {}
            Err(error) => failure = Some(error),
        }
    }
    drop(stream);
    for _ in 0..64 {
        tokio::task::yield_now().await;
    }
    match failure {
        Some(error) => Err(error),
        None => Ok(output.expect("a final response")),
    }
}

fn cancelled_reason(error: &rig::agent::StreamingError) -> &str {
    match error {
        rig::agent::StreamingError::Prompt(error) => match &**error {
            PromptError::PromptCancelled { reason, .. } => reason,
            other => panic!("a cancelled run, not {other:?}"),
        },
        other => panic!("a cancelled run, not {other:?}"),
    }
}

/// The medium's baseline: a valid call streamed as deltas, dispatched, answered.
#[tokio::test]
async fn delta_baseline_effect_log_is_the_golden_fixture() {
    let agent = AgentBuilder::new(MockCompletionModel::from_stream_turns([
        stream_turn(delta_call("call-1", "add")),
        stream_turn(vec![MockStreamEvent::text(ANSWER)]),
    ]))
    .name("golden")
    .preamble(PREAMBLE)
    .tool(Add)
    .record_effects_with_events()
    .build();
    let output = streamed(&agent, 3, UnhandledInvalidToolCall::Fail, 0)
        .await
        .expect("the run answers");
    assert_eq!(output, ANSWER);
    let log = agent.take_effect_log().expect("recording");
    assert_eq!(
        families(&log),
        [
            EffectFamily::Completion,
            EffectFamily::Tool,
            EffectFamily::Completion
        ]
    );
    assert!(log.records[0].events.is_some(), "events are kept");
    crate::goldens::golden_effects("mock_delta_baseline", &log);
}

/// An unknown name delta, retried once with feedback.
#[tokio::test]
async fn delta_retry_effect_log_is_the_golden_fixture() {
    let agent = AgentBuilder::new(MockCompletionModel::from_stream_turns([
        stream_turn(delta_call("call-1", "multiply")),
        stream_turn(delta_call("call-2", "add")),
        stream_turn(vec![MockStreamEvent::text(ANSWER)]),
    ]))
    .name("golden")
    .preamble(PREAMBLE)
    .tool(Add)
    .add_hook(RetryUnknownTool)
    .record_effects_with_events()
    .build();
    let output = streamed(&agent, 4, UnhandledInvalidToolCall::Fail, 1)
        .await
        .expect("the retry recovers");
    assert_eq!(output, ANSWER);
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
    crate::goldens::golden_effects("mock_delta_retry", &log);
}

/// The unknown call's arguments arrive before its name: buffered, then
/// surfaced with the name, then retried.
#[tokio::test]
async fn delta_retry_arguments_first_effect_log_is_the_golden_fixture() {
    let agent = AgentBuilder::new(MockCompletionModel::from_stream_turns([
        stream_turn(delta_call_arguments_first("call-1", "multiply")),
        stream_turn(delta_call_arguments_first("call-2", "add")),
        stream_turn(vec![MockStreamEvent::text(ANSWER)]),
    ]))
    .name("golden")
    .preamble(PREAMBLE)
    .tool(Add)
    .add_hook(RetryUnknownTool)
    .record_effects_with_events()
    .build();
    let output = streamed(&agent, 4, UnhandledInvalidToolCall::Fail, 1)
        .await
        .expect("the retry recovers");
    assert_eq!(output, ANSWER);
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
    crate::goldens::golden_effects("mock_delta_retry_arguments_first", &log);
}

/// An unknown name delta repaired to `add`: the arguments that follow
/// stream under the repaired name and the call runs.
#[tokio::test]
async fn delta_repair_effect_log_is_the_golden_fixture() {
    let agent = AgentBuilder::new(MockCompletionModel::from_stream_turns([
        stream_turn(delta_call("call-1", "multiply")),
        stream_turn(vec![MockStreamEvent::text(ANSWER)]),
    ]))
    .name("golden")
    .preamble(PREAMBLE)
    .tool(Add)
    .add_hook(RepairToAdd)
    .record_effects_with_events()
    .build();
    let output = streamed(&agent, 3, UnhandledInvalidToolCall::Fail, 0)
        .await
        .expect("the repair recovers");
    assert_eq!(output, ANSWER);
    let log = agent.take_effect_log().expect("recording");
    assert_eq!(
        families(&log),
        [
            EffectFamily::Completion,
            EffectFamily::Tool,
            EffectFamily::Completion
        ]
    );
    crate::goldens::golden_effects("mock_delta_repair", &log);
}

/// An unknown name delta skipped with a reason: the turn is abandoned,
/// the reason is in the transcript, the next turn answers.
#[tokio::test]
async fn delta_skip_effect_log_is_the_golden_fixture() {
    let agent = AgentBuilder::new(MockCompletionModel::from_stream_turns([
        stream_turn(delta_call("call-1", "multiply")),
        stream_turn(vec![MockStreamEvent::text(ANSWER)]),
    ]))
    .name("golden")
    .preamble(PREAMBLE)
    .tool(Add)
    .add_hook(SkipUnknown)
    .record_effects_with_events()
    .build();
    let output = streamed(&agent, 3, UnhandledInvalidToolCall::Fail, 0)
        .await
        .expect("the skip recovers");
    assert_eq!(output, ANSWER);
    let log = agent.take_effect_log().expect("recording");
    assert_eq!(
        families(&log),
        [EffectFamily::Completion, EffectFamily::Completion]
    );
    crate::goldens::golden_effects("mock_delta_skip", &log);
}

/// An unknown name delta under `Ignore`, its arguments and its block end
/// following: swallowed; the ignored-only turn is the empty answer.
#[tokio::test]
async fn delta_ignore_effect_log_is_the_golden_fixture() {
    let agent = AgentBuilder::new(MockCompletionModel::from_stream_turns([
        stream_turn(delta_call("call-1", "multiply")),
        stream_turn(vec![MockStreamEvent::text(ANSWER)]),
    ]))
    .name("golden")
    .preamble(PREAMBLE)
    .tool(Add)
    .record_effects_with_events()
    .build();
    let output = streamed(&agent, 3, UnhandledInvalidToolCall::Ignore, 0)
        .await
        .expect("the ignored call does not fail the run");
    assert_eq!(output, "", "an ignored-only turn is an empty answer");
    let log = agent.take_effect_log().expect("recording");
    assert_eq!(families(&log), [EffectFamily::Completion]);
    crate::goldens::golden_effects("mock_delta_ignore", &log);
}

/// An ignored name-delta call beside a valid one in the same turn: the
/// ignored block's deltas and end are swallowed, `add` runs.
#[tokio::test]
async fn delta_ignore_beside_valid_effect_log_is_the_golden_fixture() {
    let mut turn = delta_call("call-1", "multiply");
    turn.extend(delta_call("call-2", "add"));
    let agent = AgentBuilder::new(MockCompletionModel::from_stream_turns([
        stream_turn(turn),
        stream_turn(vec![MockStreamEvent::text(ANSWER)]),
    ]))
    .name("golden")
    .preamble(PREAMBLE)
    .tool(Add)
    .record_effects_with_events()
    .build();
    let output = streamed(&agent, 3, UnhandledInvalidToolCall::Ignore, 0)
        .await
        .expect("the ignored call does not fail the run");
    assert_eq!(output, ANSWER);
    let log = agent.take_effect_log().expect("recording");
    assert_eq!(
        families(&log),
        [
            EffectFamily::Completion,
            EffectFamily::Tool,
            EffectFamily::Completion
        ]
    );
    crate::goldens::golden_effects("mock_delta_ignore_beside_valid", &log);
}

/// An unknown name delta unresolved under `Fail`: the run fails at the
/// completion record.
#[tokio::test]
async fn delta_fail_effect_log_is_the_golden_fixture() {
    let agent = AgentBuilder::new(MockCompletionModel::from_stream_turns([
        stream_turn(delta_call("call-1", "multiply")),
        stream_turn(vec![MockStreamEvent::text(ANSWER)]),
    ]))
    .name("golden")
    .preamble(PREAMBLE)
    .tool(Add)
    .record_effects_with_events()
    .build();
    let error = streamed(&agent, 3, UnhandledInvalidToolCall::Fail, 0)
        .await
        .expect_err("the unknown call fails the run");
    assert!(
        matches!(&error, rig::agent::StreamingError::Prompt(error) if matches!(**error, PromptError::UnknownToolCall { .. })),
        "{error:?}"
    );
    let log = agent.take_effect_log().expect("recording");
    assert_eq!(families(&log), [EffectFamily::Completion]);
    crate::goldens::golden_effects("mock_delta_fail", &log);
}

/// The output tool arriving as deltas: its name validates, its arguments
/// stream, the run finalizes on the assembled call without a dispatch.
#[tokio::test]
async fn delta_output_tool_effect_log_is_the_golden_fixture() {
    let agent = AgentBuilder::new(MockCompletionModel::from_stream_turns([stream_turn(vec![
        MockStreamEvent::tool_call_name_delta("call-1", "final_result"),
        MockStreamEvent::tool_call_arguments_delta(
            "call-1",
            "{\"title\": \"Seattle Rust Meetup\", \"category\": \"Technology\", ",
        ),
        MockStreamEvent::tool_call_arguments_delta("call-1", "\"summary\": \"A meetup.\"}"),
        MockStreamEvent::tool_call_end("call-1"),
    ])]))
    .name("golden")
    .preamble("You are a concise assistant. Answer directly.")
    .output_schema_raw(event_schema())
    .output_mode(OutputMode::Tool)
    .record_effects_with_events()
    .build();
    let mut stream = agent
        .stream_prompt("Return a concise event object for a local Rust meetup in Seattle.")
        .stream()
        .await;
    let mut output = None;
    while let Some(item) = stream.next().await {
        if let MultiTurnStreamItem::FinalResponse(response) = item.expect("the stream yields") {
            output = Some(response.output);
        }
    }
    drop(stream);
    let output = output.expect("a final response");
    assert_eq!(
        output,
        json!({"title": "Seattle Rust Meetup", "category": "Technology", "summary": "A meetup."})
            .to_string()
    );
    let log = agent.take_effect_log().expect("recording");
    assert_eq!(families(&log), [EffectFamily::Completion]);
    crate::goldens::golden_effects("mock_delta_output_tool", &log);
}

/// A stop on the delta that names the tool.
#[tokio::test]
async fn delta_stop_on_name_effect_log_is_the_golden_fixture() {
    let agent = AgentBuilder::new(MockCompletionModel::from_stream_turns([stream_turn(
        delta_call("call-1", "add"),
    )]))
    .name("golden")
    .preamble(PREAMBLE)
    .tool(Add)
    .add_hook(StopOnToolNameDelta)
    .record_effects_with_events()
    .build();
    let error = streamed(&agent, 3, UnhandledInvalidToolCall::Fail, 0)
        .await
        .expect_err("the hook stops the run");
    assert_eq!(cancelled_reason(&error), STOP_ON_TOOL_NAME_DELTA);
    let log = agent.take_effect_log().expect("recording");
    assert_eq!(families(&log), [EffectFamily::Completion]);
    assert!(
        log.records[0].outcome.is_ok()
            || matches!(&log.records[0].outcome, Err(report) if report.kind == ErrorKind::Cancelled),
        "{:?}",
        log.records[0].outcome
    );
    crate::goldens::golden_effects("mock_delta_stop_on_name", &log);
}

/// A stop on the first arguments delta, after the name validated.
#[tokio::test]
async fn delta_stop_on_arguments_effect_log_is_the_golden_fixture() {
    let agent = AgentBuilder::new(MockCompletionModel::from_stream_turns([stream_turn(
        delta_call("call-1", "add"),
    )]))
    .name("golden")
    .preamble(PREAMBLE)
    .tool(Add)
    .add_hook(StopOnToolArgumentsDelta)
    .record_effects_with_events()
    .build();
    let error = streamed(&agent, 3, UnhandledInvalidToolCall::Fail, 0)
        .await
        .expect_err("the hook stops the run");
    assert_eq!(cancelled_reason(&error), STOP_ON_TOOL_ARGUMENTS_DELTA);
    let log = agent.take_effect_log().expect("recording");
    assert_eq!(families(&log), [EffectFamily::Completion]);
    crate::goldens::golden_effects("mock_delta_stop_on_arguments", &log);
}
