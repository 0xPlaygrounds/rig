//! Golden `mock_outcome_invalid_call_unhandled` (Matrix D): a model calls
//! a tool the program does not advertise and no hook resolves it, so the
//! run fails at that record with `PromptError::UnknownToolCall`
//! (`UnhandledInvalidToolCall::Fail`, the default). One completion is
//! recorded; the failure is the run's, not the record's.
//!
//! Scripted from the mock model: no live model in the corpus emits a call
//! to a tool that is not in its request.

use rig::agent::AgentBuilder;
use rig::completion::PromptError;
use rig::effect::EffectFamily;
use rig::test_utils::{MockCompletionModel, MockTurn};
use serde_json::json;

use super::golden_recovery::Add;
use crate::goldens::families;

#[tokio::test]
async fn outcome_invalid_call_unhandled_effect_log_is_the_golden_fixture() {
    let agent = AgentBuilder::new(MockCompletionModel::from_turns([MockTurn::tool_call(
        "call-1",
        "multiply",
        json!({"x": 2, "y": 3}),
    )]))
    .name("golden")
    .preamble("Use the add tool.")
    .tool(Add)
    .record_effects()
    .build();
    let error = agent
        .prompt("What is 2 * 3?")
        .max_turns(3)
        .await
        .expect_err("an unknown tool with no hook fails the run");
    assert!(
        matches!(error, PromptError::UnknownToolCall { ref tool_name, .. } if tool_name == "multiply"),
        "{error:?}"
    );
    let log = agent.take_effect_log().expect("recording");
    assert_eq!(families(&log), [EffectFamily::Completion]);
    crate::goldens::golden_effects("mock_outcome_invalid_call_unhandled", &log);
}
