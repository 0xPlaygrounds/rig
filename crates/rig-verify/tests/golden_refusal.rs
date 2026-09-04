//! A stale golden refuses; it never passes with a different trace.
//!
//! The header names the program: a golden whose run spec, hook stack,
//! required row or bus policy differs from the agent's is refused before
//! the first dispatch, with both sides in the message. A golden whose
//! *record* differs — one field of one recorded request — fails the run at
//! that record, naming the JSON pointer of the difference, and the run
//! never reaches the record after it.

#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

use std::time::Duration;

use rig_agent::{AgentBuilder, run::PromptError};
use rig_bus::Bus;
use rig_core::{
    effect::{EffectFamily, EffectKind, HandlerKey},
    error::ErrorKind,
};
use rig_effect_log::{EffectLog, EffectLogRecorder, EffectLogReplayer};

/// The root suite's constants, verbatim (`tests/common/support.rs`).
const PREAMBLE: &str = "You are a concise assistant. Answer directly.";
const PROMPT: &str = "In one or two sentences, explain what Rust programming language is and why memory safety matters.";
const OWNER: &str = "golden";
const CONVERSATION: &str = "golden-conversation";

fn golden(fixture: &str) -> EffectLog {
    let path = format!(
        "{}/fixtures/{fixture}.effects.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).expect("the golden fixture is committed");
    serde_json::from_str(&text).expect("the golden fixture loads")
}

fn model_key() -> HandlerKey {
    HandlerKey::from(format!("{OWNER}/model:default"))
}

fn memory_key() -> HandlerKey {
    HandlerKey::from(format!("{OWNER}/memory"))
}

/// The memory-conversation program over a bus whose model and memory are
/// replayers of `log`, with a recorder on the driver.
struct Replay {
    agent: rig_agent::Agent,
    recorder: EffectLogRecorder,
    driver: tokio::task::JoinHandle<()>,
}

fn memory_conversation_over(log: &EffectLog) -> Replay {
    let (dispatcher, registrar, mut driver) =
        Bus::channel_with(log.header.bus.expect("the header names the bus policy"));
    let model = EffectLogReplayer::for_key(log, &model_key()).expect("the model's records");
    driver
        .register_erased(model_key(), rig_core::serve::ErasedHandler::new(model))
        .expect("a fresh key");
    let recorder = EffectLogRecorder::new();
    driver.record_to(recorder.clone());
    let driver = tokio::spawn(driver);
    let memory =
        EffectLogReplayer::for_key(log, &memory_key()).expect("the conversation's records");
    let agent = AgentBuilder::over_bus(dispatcher, registrar, OWNER, model_key())
        .name(OWNER)
        .preamble(PREAMBLE)
        .memory_handler(memory)
        .conversation(CONVERSATION)
        .build();
    Replay {
        agent,
        recorder,
        driver,
    }
}

fn refusal(log: &EffectLog) -> String {
    let replay = memory_conversation_over(log);
    let report = replay
        .agent
        .check_replayable(log)
        .expect_err("a stale golden is refused");
    drop(replay.agent);
    report.message.to_owned()
}

#[test]
fn the_golden_loads_and_is_the_program() {
    let log = golden("anthropic_memory_conversation");
    EffectLogReplayer::check_header(&log).expect("a current format");
    assert_eq!(
        log.iter()
            .map(|record| record.kind.family())
            .collect::<Vec<_>>(),
        [
            EffectFamily::Memory,
            EffectFamily::Completion,
            EffectFamily::Memory
        ]
    );
    let runtime = tokio::runtime::Runtime::new().expect("a runtime");
    runtime.block_on(async {
        let replay = memory_conversation_over(&log);
        replay
            .agent
            .check_replayable(&log)
            .expect("the golden is this program's");
        drop(replay.agent);
        drop(replay.recorder);
        replay.driver.await.expect("driver task");
    });
}

#[tokio::test]
async fn a_golden_of_another_run_spec_is_refused_with_both_hashes() {
    let mut log = golden("anthropic_memory_conversation");
    let recorded = log.header.run_spec.expect("the golden names its run spec");
    log.header.run_spec = Some(recorded ^ 1);
    let message = refusal(&log);
    assert!(
        message.contains(&format!("{:#018x}", recorded ^ 1))
            && message.contains(&format!("{recorded:#018x}")),
        "both hashes are named: {message}"
    );
}

#[tokio::test]
async fn a_golden_of_another_hook_stack_is_refused_with_both_stacks() {
    let mut log = golden("anthropic_memory_conversation");
    log.header.hooks.push("Ghost".to_owned());
    let message = refusal(&log);
    assert!(
        message.contains("[\"Ghost\"]") && message.contains("runs under []"),
        "both stacks are named: {message}"
    );
}

#[tokio::test]
async fn a_golden_of_another_required_row_is_refused_with_both_rows() {
    let mut log = golden("anthropic_memory_conversation");
    let ghost = HandlerKey::from(format!("{OWNER}/tool:ghost#0"));
    log.header
        .required
        .insert(ghost.clone(), EffectFamily::Tool);
    let message = refusal(&log);
    assert!(
        message.contains("tool:ghost#0") && message.contains("this agent requires"),
        "both rows are named: {message}"
    );
}

/// A bus policy is checked by an agent that owns its bus (one built over a
/// host's bus does not know the host's policy). The agent here owns one
/// with the recorded policy; the golden claims another.
#[tokio::test]
async fn a_golden_of_another_bus_policy_is_refused_with_both_policies() {
    let mut log = golden("anthropic_memory_conversation");
    let recorded = log.header.bus.expect("the header names the bus policy");
    let other = rig_core::serve::ServingPolicy {
        serial_per_handler: !recorded.serial_per_handler,
        ..recorded
    };
    log.header.bus = Some(other);
    let memory =
        EffectLogReplayer::for_key(&log, &memory_key()).expect("the conversation's records");
    let agent = AgentBuilder::new(rig_core::test_utils::MockCompletionModel::from_turns([]))
        .name(OWNER)
        .preamble(PREAMBLE)
        .configure_bus(recorded)
        .memory_handler(memory)
        .conversation(CONVERSATION)
        .build();
    let report = agent
        .check_replayable(&log)
        .expect_err("another bus policy is refused");
    let message = report.message;
    assert!(
        message.contains(&format!("{other:?}")) && message.contains(&format!("{recorded:?}")),
        "both policies are named: {message}"
    );
}

#[tokio::test]
async fn a_golden_whose_recorded_request_differs_fails_at_that_record_and_no_later() {
    let mut log = golden("anthropic_memory_conversation");
    match &mut log.records[1].kind {
        EffectKind::Completion { request, .. } => {
            request.temperature = Some(0.25);
        }
        other => panic!("record 1 is the completion: {other:?}"),
    }
    let replay = memory_conversation_over(&log);
    let error = tokio::time::timeout(Duration::from_secs(5), replay.agent.prompt(PROMPT).run())
        .await
        .expect("a replay never hangs")
        .expect_err("a divergent record fails the run");
    let PromptError::Report(report) = &error else {
        panic!("the divergence is reported: {error:?}");
    };
    assert_eq!(report.kind, ErrorKind::Divergence);
    assert!(
        report.message.contains("payload.request.temperature"),
        "the pointer names the field: {}",
        report.message
    );
    drop(replay.agent);
    replay.driver.await.expect("driver task");
    let replayed = replay.recorder.take();
    assert_eq!(
        replayed
            .iter()
            .map(|record| (record.kind.family(), record.outcome.is_ok()))
            .collect::<Vec<_>>(),
        [
            (EffectFamily::Memory, true),
            (EffectFamily::Completion, false)
        ],
        "the load, then the divergence; never the append"
    );
}
