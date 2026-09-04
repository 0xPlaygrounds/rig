//! Matrix P's mock-scripted cells: a suspending layer approving, denying
//! and cancelled mid-suspend; a patch of the wrong family; an error
//! replacing a streamed answer. Scripted because the world's timing and a
//! stream's cancellation are the cell, not the wire; the layers are in
//! `tests/common/goldens.rs`. The enumeration and the replays live in
//! `crates/rig-verify/tests/corpus_layers.rs`.

use std::time::Duration;

use futures::StreamExt;
use rig::agent::{AgentBuilder, MultiTurnStreamItem, StreamingError};
use rig::effect::EffectFamily;
use rig::serve::ErasedHandler;
use rig::test_utils::{MockCompletionModel, MockStreamEvent, MockTurn};
use rig_effect_log::EffectLog;
use serde_json::json;

use crate::goldens::{
    Answer, ApprovalLayer, CANCEL_STREAM_REASON, CancelStreamLayer, WORLD_DENY_REASON,
    WrongFamilyLayer, add_tool_under, families, spawn_world,
};

const TOOLS_PREAMBLE: &str = "You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.";
const BASIC_PREAMBLE: &str = "You are a concise assistant. Answer directly.";
const ADD_PROMPT: &str = "Use the add tool to add 17 and 25, then reply with just the number.";

fn calls_add_then_answers() -> MockCompletionModel {
    MockCompletionModel::from_turns([
        MockTurn::tool_call("call-1", "add", json!({"x": 17, "y": 25})),
        MockTurn::text("42"),
    ])
}

/// The text of every tool result in the last request's history.
fn tool_result_texts(log: &EffectLog) -> Vec<String> {
    let request = match &log.records.last().expect("a record").kind {
        rig::effect::EffectKind::Completion { request, .. } => request,
        other => panic!("the last record is a completion, not {other:?}"),
    };
    request
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
                        other => format!("{other:?}"),
                    })
                    .collect::<String>(),
            ),
            _ => None,
        })
        .collect()
}

/// The program under a suspending layer on `add`, the world answering as
/// `answer` says; on `Never` the run is dropped once the world was asked.
async fn suspended(answer: Answer) -> EffectLog {
    let reached = std::sync::Arc::new(tokio::sync::Notify::new());
    let asks = spawn_world(answer, std::sync::Arc::clone(&reached));
    let server = add_tool_under(|adder| adder.layered(ApprovalLayer { asks }));
    let agent = AgentBuilder::new(calls_add_then_answers())
        .name("golden")
        .preamble(TOOLS_PREAMBLE)
        .temperature(0.0)
        .tool_server_handle(server)
        .record_effects()
        .build();
    if answer == Answer::Never {
        let run = agent.prompt(ADD_PROMPT).max_turns(3);
        tokio::select! {
            finished = tokio::time::timeout(Duration::from_secs(5), run) => {
                panic!("the run finished before the world was asked: {finished:?}")
            }
            asked = tokio::time::timeout(Duration::from_secs(5), reached.notified()) => {
                asked.expect("the world is asked within the guard");
            }
        }
        for _ in 0..200 {
            tokio::task::yield_now().await;
        }
    } else {
        let response = tokio::time::timeout(
            Duration::from_secs(5),
            agent.prompt(ADD_PROMPT).max_turns(3),
        )
        .await
        .expect("a suspended layer never hangs the run past its answer")
        .expect("the agent answers");
        assert_eq!(response.output, "42");
    }
    agent.take_effect_log().expect("recording")
}

#[tokio::test]
async fn suspend_approve_effect_log_is_the_golden_fixture() {
    let log = suspended(Answer::Approve).await;
    assert_eq!(
        families(&log),
        [
            EffectFamily::Completion,
            EffectFamily::Tool,
            EffectFamily::Completion
        ]
    );
    assert_eq!(log.header.hooks, ["ApprovalLayer"]);
    crate::goldens::golden_effects("mock_layers_suspend_approve", &log);
}

#[tokio::test]
async fn suspend_deny_effect_log_is_the_golden_fixture() {
    let log = suspended(Answer::Deny).await;
    // No record for the denial; the model saw the skipped result.
    assert_eq!(
        families(&log),
        [EffectFamily::Completion, EffectFamily::Completion]
    );
    assert_eq!(tool_result_texts(&log), [WORLD_DENY_REASON]);
    crate::goldens::golden_effects("mock_layers_suspend_deny", &log);
}

#[tokio::test]
async fn suspend_cancelled_effect_log_is_the_golden_fixture() {
    let log = suspended(Answer::Never).await;
    // The consumer's drop mid-suspend: the tool record says `Cancelled`
    // (from the drop, not the layer); the run never answered.
    assert_eq!(
        families(&log),
        [EffectFamily::Completion, EffectFamily::Tool]
    );
    assert!(
        matches!(&log.records[1].outcome, Err(report) if report.kind == rig::error::ErrorKind::Cancelled),
        "{:?}",
        log.records[1].outcome
    );
    crate::goldens::golden_effects("mock_layers_suspend_cancelled", &log);
}

#[tokio::test]
async fn wrong_family_patch_effect_log_is_the_golden_fixture() {
    let server = add_tool_under(|adder| adder.layered(WrongFamilyLayer));
    let agent = AgentBuilder::new(calls_add_then_answers())
        .name("golden")
        .preamble(TOOLS_PREAMBLE)
        .temperature(0.0)
        .tool_server_handle(server)
        .record_effects()
        .build();
    let response = agent
        .prompt(ADD_PROMPT)
        .max_turns(3)
        .await
        .expect("the run goes on: the tool failed, the model answered");
    assert_eq!(response.output, "42");
    let log = agent.take_effect_log().expect("recording");
    // `Internal`, no record: the engine turns the tool's failure into a
    // failed result the model sees, and the run goes on.
    assert_eq!(
        families(&log),
        [EffectFamily::Completion, EffectFamily::Completion]
    );
    let texts = tool_result_texts(&log);
    assert_eq!(texts.len(), 1);
    assert!(
        texts[0].contains("layer `WrongFamilyLayer`")
            && texts[0].contains("never changes the family"),
        "{texts:?}"
    );
    crate::goldens::golden_effects("mock_layers_wrong_family_patch", &log);
}

#[tokio::test]
async fn replace_streamed_cancelled_effect_log_is_the_golden_fixture() {
    // A layer on the model key, over a host bus, replaces the streamed
    // answer with a cancel in `after`: the events were delivered, the
    // record holds the real answer and its events, and the run ends
    // cancelled.
    let model = MockCompletionModel::from_stream_turns([vec![
        MockStreamEvent::text("stre"),
        MockStreamEvent::text("amed"),
        MockStreamEvent::final_response_with_total_tokens(2),
    ]]);
    let (dispatcher, registrar, mut driver) = rig::bus::Bus::channel();
    let model_key = rig::effect::HandlerKey::from("golden/model:default");
    driver
        .register_erased(
            model_key.clone(),
            ErasedHandler::new(rig::serve::adapters::CompletionAdapter::new(
                "default", model,
            ))
            .layered(CancelStreamLayer),
        )
        .expect("a fresh key");
    let recorder = rig_effect_log::EffectLogRecorder::keeping_stream_events();
    driver.record_to(recorder.clone());
    let driver = tokio::spawn(driver);
    let agent = AgentBuilder::over_bus(dispatcher.clone(), registrar.clone(), "golden", model_key)
        .name("golden")
        .preamble(BASIC_PREAMBLE)
        .temperature(0.0)
        .build();
    let mut stream = agent
        .stream_prompt("Reply with the single word: ready.")
        .stream()
        .await;
    let mut ending = None;
    let mut texts = String::new();
    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::StreamAssistantItem(
                rig::streaming::StreamEvent::BlockDelta {
                    delta: rig::streaming::Delta::Text { text },
                    ..
                },
            )) => {
                texts.push_str(&text);
            }
            Ok(_) => {}
            Err(error) => {
                ending = Some(error);
                break;
            }
        }
    }
    drop(stream);
    let ending = ending.expect("the run ends in an error");
    match &ending {
        StreamingError::Report(report) => {
            assert_eq!(report.kind, rig::error::ErrorKind::Cancelled);
            assert_eq!(report.message, CANCEL_STREAM_REASON);
        }
        StreamingError::Prompt(error) => {
            assert!(
                matches!(&**error, rig::completion::PromptError::PromptCancelled { reason, .. } if reason == CANCEL_STREAM_REASON),
                "{error:?}"
            );
        }
        other => panic!("a cancel, not {other:?}"),
    }
    assert_eq!(texts, "streamed", "the events were delivered as they came");
    for _ in 0..64 {
        tokio::task::yield_now().await;
    }
    let log = agent.stamp(recorder.take());
    drop((agent, dispatcher, registrar));
    driver.await.expect("the host's driver");
    assert_eq!(families(&log), [EffectFamily::Completion]);
    assert!(
        log.records[0].outcome.is_ok(),
        "the record holds the real answer"
    );
    assert!(log.records[0].events.is_some(), "and its events");
    assert_eq!(log.header.hooks, ["CancelStreamLayer"]);
    crate::goldens::golden_effects("mock_layers_replace_streamed_cancelled", &log);
}
