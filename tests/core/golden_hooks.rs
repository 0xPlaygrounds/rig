//! Golden `mock_hooks_retry_twice` (Matrix B): a model calls a tool the
//! program does not advertise twice in a row; the run retries twice under
//! `max_invalid_tool_call_retries(2)` and the `RetryUnknownTool` hook,
//! then the model calls the real tool and answers. Two retries are two
//! more completion records; the policy is program.
//!
//! Scripted from the mock model: no live model in the corpus emits a call
//! to a tool that is not in its request (the corpus follow-up tried twice
//! with Sonnet 4.6 and stopped).

use rig::agent::AgentBuilder;
use rig::effect::EffectFamily;
use rig::test_utils::{MockCompletionModel, MockTurn};
use serde_json::json;

use super::golden_recovery::Add;
use crate::goldens::families;

pub(crate) fn script() -> MockCompletionModel {
    MockCompletionModel::from_turns([
        MockTurn::tool_call("call-1", "multiply", json!({"x": 2, "y": 3})),
        MockTurn::tool_call("call-2", "divide", json!({"x": 2, "y": 3})),
        MockTurn::tool_call("call-3", "add", json!({"x": 2, "y": 3})),
        MockTurn::text("2 + 3 = 5"),
    ])
}

#[tokio::test]
async fn hooks_retry_twice_effect_log_is_the_golden_fixture() {
    let agent = AgentBuilder::new(script())
        .name("golden")
        .preamble("Use the add tool.")
        .tool(Add)
        .add_hook(crate::goldens::RetryUnknownTool)
        .record_effects()
        .build();
    let response = agent
        .prompt("What is 2 + 3?")
        .max_turns(5)
        .max_invalid_tool_call_retries(2)
        .await
        .expect("the second retry recovers");
    assert_eq!(response.output, "2 + 3 = 5");
    let log = agent.take_effect_log().expect("recording");
    assert_eq!(
        families(&log),
        [
            EffectFamily::Completion,
            EffectFamily::Completion,
            EffectFamily::Completion,
            EffectFamily::Tool,
            EffectFamily::Completion,
        ],
    );
    crate::goldens::golden_effects("mock_hooks_retry_twice", &log);
}
