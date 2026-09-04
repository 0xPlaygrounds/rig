//! Matrix T's `Denied` cells: a layer's denial under every consumer
//! mapping — on a tool (the model sees the skipped result), on a
//! completion (the run fails), on a memory `Load` (the run fails at the
//! record), on a custom effect from a hook (the hook sees `Denied`).
//! Mock-scripted: the denial is the cell, not the wire. The enumeration
//! and the replays live in `crates/rig-verify/tests/corpus_leftovers.rs`.

use rig::agent::AgentBuilder;
use rig::bus::Bus;
use rig::effect::{EffectFamily, HandlerKey};
use rig::serve::ErasedHandler;
use rig::test_utils::{MockCompletionModel, MockTurn};
use rig_effect_log::{EffectLog, EffectLogRecorder};
use serde_json::json;

use crate::goldens::{
    CONVERSATION, DenyAllLayer, HOST_DENY_REASON, NOTE_KEY, NoteDeniedAtStart, NoteTaker,
    add_tool_under, families,
};

const TOOLS_PREAMBLE: &str = "You are a calculator here to help the user perform arithmetic operations. Use the tools provided to answer the user's question.";
const BASIC_PREAMBLE: &str = "You are a concise assistant. Answer directly.";
const ADD_PROMPT: &str = "Use the add tool to add 17 and 25, then reply with just the number.";
const PROMPT: &str = "Reply with the single word: ready.";

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

#[tokio::test]
async fn denied_tool_effect_log_is_the_golden_fixture() {
    let server = add_tool_under(|adder| adder.layered(DenyAllLayer));
    let agent = AgentBuilder::new(MockCompletionModel::from_turns([
        MockTurn::tool_call("call-1", "add", json!({"x": 17, "y": 25})),
        MockTurn::text("I could not add them."),
    ]))
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
        .expect("the model sees the skipped result and answers");
    assert_eq!(response.output, "I could not add them.");
    let log = agent.take_effect_log().expect("recording");
    assert_eq!(
        families(&log),
        [EffectFamily::Completion, EffectFamily::Completion]
    );
    assert_eq!(tool_result_texts(&log), [HOST_DENY_REASON]);
    assert_eq!(log.header.hooks, ["DenyAllLayer"]);
    crate::goldens::golden_effects("mock_leftovers_denied_tool", &log);
}

#[tokio::test]
async fn denied_completion_effect_log_is_the_golden_fixture() {
    // A host's layer on the model key denies: the run fails with the
    // report, and the log holds no record.
    let (dispatcher, registrar, mut driver) = Bus::channel();
    let model_key = HandlerKey::from("golden/model:default");
    driver
        .register_erased(
            model_key.clone(),
            ErasedHandler::new(rig::serve::adapters::CompletionAdapter::new(
                "default",
                MockCompletionModel::text("never asked"),
            ))
            .layered(DenyAllLayer),
        )
        .expect("a fresh key");
    let recorder = EffectLogRecorder::new();
    driver.record_to(recorder.clone());
    let driver = tokio::spawn(driver);
    let agent = AgentBuilder::over_bus(dispatcher.clone(), registrar.clone(), "golden", model_key)
        .name("golden")
        .preamble(BASIC_PREAMBLE)
        .temperature(0.0)
        .build();
    let error = agent
        .prompt(PROMPT)
        .await
        .expect_err("the denial fails the run");
    assert!(
        matches!(&error, rig::completion::PromptError::Report(report) if report.kind == rig::error::ErrorKind::Denied),
        "{error:?}"
    );
    let log = agent.stamp(recorder.take());
    drop((agent, dispatcher, registrar));
    driver.await.expect("the host's driver");
    assert!(log.records.is_empty(), "no record for a denied completion");
    assert_eq!(log.header.hooks, ["DenyAllLayer"]);
    crate::goldens::golden_effects("mock_leftovers_denied_completion", &log);
}

#[tokio::test]
async fn denied_memory_load_effect_log_is_the_golden_fixture() {
    let memory = ErasedHandler::new(rig::serve::adapters::MemoryAdapter::new(
        rig::memory::InMemoryConversationMemory::new(),
    ))
    .layered(DenyAllLayer);
    let agent = AgentBuilder::new(MockCompletionModel::text("never asked"))
        .name("golden")
        .preamble(BASIC_PREAMBLE)
        .temperature(0.0)
        .memory_handler(memory)
        .conversation(CONVERSATION)
        .record_effects()
        .build();
    let error = agent
        .prompt(PROMPT)
        .await
        .expect_err("the denied load fails the run");
    assert!(
        matches!(
            &error,
            rig::completion::PromptError::MemoryError(rig::memory::MemoryError::Policy(reason))
                if reason == HOST_DENY_REASON
        ),
        "{error:?}"
    );
    let log = agent.take_effect_log().expect("recording");
    assert!(log.records.is_empty(), "no record for a denied load");
    assert_eq!(log.header.hooks, ["DenyAllLayer"]);
    crate::goldens::golden_effects("mock_leftovers_denied_memory_load", &log);
}

#[tokio::test]
async fn denied_custom_from_hook_effect_log_is_the_golden_fixture() {
    // The host's layer on its own note key denies the hook's note: the
    // hook sees `Denied` and the run goes on; the log holds the completion
    // only.
    let (dispatcher, registrar, mut driver) = Bus::channel();
    let model_key = HandlerKey::from("golden/model:default");
    driver
        .register_erased(
            model_key.clone(),
            ErasedHandler::new(rig::serve::adapters::CompletionAdapter::new(
                "default",
                MockCompletionModel::text("ready"),
            )),
        )
        .expect("a fresh key");
    driver
        .register_erased(
            HandlerKey::from(NOTE_KEY),
            ErasedHandler::new(NoteTaker).layered(DenyAllLayer),
        )
        .expect("a fresh key");
    let recorder = EffectLogRecorder::new();
    driver.record_to(recorder.clone());
    let driver = tokio::spawn(driver);
    let agent = AgentBuilder::over_bus(dispatcher.clone(), registrar.clone(), "golden", model_key)
        .name("golden")
        .preamble(BASIC_PREAMBLE)
        .temperature(0.0)
        .add_hook(NoteDeniedAtStart)
        .build();
    let response = agent.prompt(PROMPT).await.expect("the run goes on");
    assert_eq!(response.output, "ready");
    let log = agent.stamp(recorder.take());
    drop((agent, dispatcher, registrar));
    driver.await.expect("the host's driver");
    assert_eq!(families(&log), [EffectFamily::Completion]);
    assert_eq!(log.header.hooks, ["NoteDeniedAtStart", "DenyAllLayer"]);
    crate::goldens::golden_effects("mock_leftovers_denied_custom_from_hook", &log);
}
