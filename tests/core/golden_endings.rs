//! Matrix F's no-record cells: a hook that stops the run before any
//! dispatch — at run start, at model selection, before the completion
//! call. Nothing reaches the wire, so no cassette exists to record; the
//! model is a mock that is never asked, and the golden is a header over an
//! empty record. The run ends in `PromptCancelled` with the hook's reason.

use rig::agent::AgentBuilder;
use rig::completion::PromptError;
use rig::test_utils::MockCompletionModel;

use super::golden_recovery::Add;
use crate::goldens::{
    RecordSettled, STOP_AT_COMPLETION_CALL, STOP_AT_MODEL_SELECT, STOP_AT_START,
    StopAtCompletionCall, StopAtModelSelect, StopAtStart,
};

fn cancelled_reason(error: &PromptError) -> &str {
    match error {
        PromptError::PromptCancelled { reason, .. } => reason,
        other => panic!("a cancelled run, not {other:?}"),
    }
}

async fn stops_before_any_dispatch(
    hook: impl rig::agent::AgentHook + 'static,
    reason: &str,
    golden: &str,
) -> rig::effect_log::EffectLog {
    let settled = RecordSettled::default();
    let agent = AgentBuilder::new(MockCompletionModel::text("never asked"))
        .name("golden")
        .preamble("Use the add tool.")
        .tool(Add)
        .add_hook(hook)
        .add_hook(settled.clone())
        .record_effects()
        .build();
    let error = agent
        .prompt("What is 2 + 3?")
        .max_turns(3)
        .await
        .expect_err("the hook stops the run");
    assert_eq!(cancelled_reason(&error), reason);
    let seen = settled.0.lock().expect("settled").clone();
    assert!(
        seen.as_deref()
            .is_some_and(|seen| seen.starts_with("error:")),
        "on_run_settled saw the error: {seen:?}"
    );
    let log = agent.take_effect_log().expect("recording");
    assert!(log.records.is_empty(), "nothing was dispatched: {golden}");
    log
}

#[tokio::test]
async fn endings_stop_at_start_effect_log_is_the_golden_fixture() {
    let log = stops_before_any_dispatch(StopAtStart, STOP_AT_START, "start").await;
    crate::goldens::golden_effects("mock_endings_stop_at_start", &log);
}

#[tokio::test]
async fn endings_stop_at_model_select_effect_log_is_the_golden_fixture() {
    let log = stops_before_any_dispatch(StopAtModelSelect, STOP_AT_MODEL_SELECT, "select").await;
    crate::goldens::golden_effects("mock_endings_stop_at_model_select", &log);
}

#[tokio::test]
async fn endings_stop_at_completion_call_effect_log_is_the_golden_fixture() {
    let log =
        stops_before_any_dispatch(StopAtCompletionCall, STOP_AT_COMPLETION_CALL, "call").await;
    crate::goldens::golden_effects("mock_endings_stop_at_completion_call", &log);
}
