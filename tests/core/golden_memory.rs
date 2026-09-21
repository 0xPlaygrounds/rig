//! Matrix J's no-wire cell: a `Load` that fails. The run fails before
//! any completion, so the model is a mock that is never asked, and the
//! golden is one memory record holding the store's error.

use rig::agent::AgentBuilder;
use rig::completion::PromptError;
use rig::effect::EffectFamily;
use rig::test_utils::MockCompletionModel;
use rig_cassette::agent::AgentReplayExt;

use crate::goldens::{CONVERSATION, FailingMemory, families};

#[tokio::test]
async fn memory_failing_load_effect_log_is_the_golden_fixture() {
    let recorder = rig_cassette::effect_log::EffectLogRecorder::new();
    let agent = AgentBuilder::new(MockCompletionModel::text("never asked"))
        .name("golden")
        .preamble("You are a concise assistant. Answer directly.")
        .memory(FailingMemory::load_fails())
        .conversation(CONVERSATION)
        .record_to(recorder.clone())
        .build();
    let error = agent
        .prompt("Reply with the single word: ready.")
        .await
        .expect_err("the load fails the run");
    assert!(
        matches!(
            &error,
            PromptError::MemoryError(rig::memory::MemoryError::Backend(_))
        ),
        "{error:?}"
    );
    let log = agent.stamp(recorder.take());
    assert_eq!(families(&log), [EffectFamily::Memory]);
    assert!(
        matches!(&log.records[0].outcome, Err(report)
            if report.kind == rig::error::ErrorKind::MemoryBackend
                && report.source_chain == ["the store refused the load"]),
        "{:?}",
        log.records[0].outcome
    );
    crate::goldens::golden_effects("mock_memory_failing_load", &log);
}
