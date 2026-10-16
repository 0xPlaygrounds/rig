//! A stale golden refuses; it never passes with a different trace.
//!
//! The header names the program: a golden whose run spec, hook stack or
//! required row differs from the agent's is refused before
//! the first dispatch, with both sides in the message. A golden whose
//! *record* differs — one field of one recorded request — fails the run at
//! that record, naming the JSON pointer of the difference, and the run
//! never reaches the record after it.

use rig_cassette::agent::AgentReplayExt;
use std::time::Duration;

use rig_agent::handlers::Handlers;
use rig_agent::{AgentBuilder, run::PromptError};
use rig_cassette::effect_log::{EffectLog, EffectLogRecorder, EffectLogReplayer};
use rig_core::{
    effect::{EffectFamily, EffectKind, HandlerKey},
    error::ErrorKind,
};

use crate::corpus::golden;

/// The root suite's constants, verbatim (`test-support/rig-test-support/src/support.rs`).
const PREAMBLE: &str = "You are a concise assistant. Answer directly.";
const PROMPT: &str = "In one or two sentences, explain what Rust programming language is and why memory safety matters.";
const OWNER: &str = "golden";
const GOLDEN: &str = "anthropic_completion_smoke";

fn model_key() -> HandlerKey {
    HandlerKey::from(format!("{OWNER}/model:default"))
}

/// The completion-smoke program over handlers whose model is a replayer
/// of `log`, with a recorder on them.
struct Replay {
    agent: rig_agent::Agent,
    recorder: EffectLogRecorder,
}

fn completion_smoke_over(log: &EffectLog) -> Replay {
    let handlers = Handlers::default();
    let model = EffectLogReplayer::for_key(log, &model_key()).expect("the model's records");
    handlers
        .register_erased(model_key(), rig_core::serve::ErasedHandler::new(model))
        .expect("a fresh key");
    let recorder = EffectLogRecorder::new();
    handlers.record_to(recorder.clone());
    let agent = AgentBuilder::over_handlers(handlers.clone(), OWNER, model_key())
        .name(OWNER)
        .preamble(PREAMBLE)
        .build();
    Replay { agent, recorder }
}

fn refusal(log: &EffectLog) -> String {
    let replay = completion_smoke_over(log);
    let report = replay
        .agent
        .check_replayable(log)
        .expect_err("a stale golden is refused");
    drop(replay.agent);
    report.message.to_owned()
}

#[test]
fn the_golden_loads_and_is_the_program() {
    let log = golden(GOLDEN);
    EffectLogReplayer::check_header(&log).expect("a current format");
    assert_eq!(
        log.iter()
            .map(|record| record.kind.family())
            .collect::<Vec<_>>(),
        [EffectFamily::Completion]
    );
    let runtime = tokio::runtime::Runtime::new().expect("a runtime");
    runtime.block_on(async {
        let replay = completion_smoke_over(&log);
        replay
            .agent
            .check_replayable(&log)
            .expect("the golden is this program's");
        drop(replay.agent);
        drop(replay.recorder);
    });
}

#[tokio::test]
async fn a_golden_of_another_run_spec_is_refused_with_both_hashes() {
    let mut log = golden(GOLDEN);
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
    let mut log = golden(GOLDEN);
    log.header.hooks.push("Ghost".to_owned());
    let message = refusal(&log);
    assert!(
        message.contains("[\"Ghost\"]") && message.contains("runs under []"),
        "both stacks are named: {message}"
    );
}

#[tokio::test]
async fn a_golden_of_another_required_row_is_refused_with_both_rows() {
    let mut log = golden(GOLDEN);
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

#[tokio::test]
async fn a_golden_whose_recorded_request_differs_fails_at_that_record_and_no_later() {
    let mut log = golden(GOLDEN);
    match &mut log.records[0].kind {
        EffectKind::Completion { request, .. } => {
            std::sync::Arc::make_mut(request).temperature = Some(0.25);
        }
        other => panic!("record 0 is the completion: {other:?}"),
    }
    let replay = completion_smoke_over(&log);
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
    let replayed = replay.recorder.take();
    assert_eq!(
        replayed
            .iter()
            .map(|record| (record.kind.family(), record.outcome.is_ok()))
            .collect::<Vec<_>>(),
        [(EffectFamily::Completion, false)],
        "the divergence, and nothing after it"
    );
}
