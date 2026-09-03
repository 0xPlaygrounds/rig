//! Golden `mock_invalid_tool_call_recovery`: a model's first turn calls a
//! tool the program does not advertise; the run retries once and the
//! model then calls the real tool and answers. The retry is a record (a
//! second completion), the policy decision (retry rather than fail) is
//! program, not record.
//!
//! No existing cassette has an engine-driven invalid call, so this golden
//! is scripted from the mock model rather than a cassette — the only one in
//! the corpus that is. It still holds the corpus contract: recorded once by
//! this producer, replayed by rig-verify with nothing behind the keys.

use rig::agent::{
    AgentBuilder, AgentHook, HookContext, InvalidToolCallAction, InvalidToolCallContext,
};
use rig::test_utils::{MockCompletionModel, MockTurn};
use rig::tool::{Tool, ToolContext, ToolExecutionError};
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize)]
struct AddArgs {
    x: i64,
    y: i64,
}

struct Add;

impl Tool for Add {
    const NAME: &'static str = "add";
    type Args = AddArgs;
    type Output = i64;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "adds two integers".into()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}, "required": ["x", "y"]})
    }

    async fn call(&self, _context: &mut ToolContext, args: AddArgs) -> Result<i64, Self::Error> {
        Ok(args.x + args.y)
    }
}

/// The program's recovery policy: an unknown tool is retried once with
/// feedback. A hook is program, not record — the golden's replay carries
/// the same hook and the header names it.
struct RetryUnknownTool;

impl AgentHook for RetryUnknownTool {
    async fn on_invalid_tool_call(
        &self,
        _ctx: &HookContext,
        context: &InvalidToolCallContext,
    ) -> Option<InvalidToolCallAction> {
        Some(InvalidToolCallAction::Retry {
            feedback: format!("there is no tool named {}; use add", context.tool_name),
        })
    }
}

pub(crate) fn script() -> MockCompletionModel {
    MockCompletionModel::from_turns([
        MockTurn::tool_call("call-1", "multiply", json!({"x": 2, "y": 3})),
        MockTurn::tool_call("call-2", "add", json!({"x": 2, "y": 3})),
        MockTurn::text("2 + 3 = 5"),
    ])
}

#[tokio::test]
async fn invalid_tool_call_recovery_effect_log_is_the_golden_fixture() {
    let agent = AgentBuilder::new(script())
        .name("golden")
        .preamble("Use the add tool.")
        .tool(Add)
        .add_hook(RetryUnknownTool)
        .record_effects()
        .build();
    let response = agent
        .prompt("What is 2 + 3?")
        .max_turns(4)
        .max_invalid_tool_call_retries(1)
        .await
        .expect("the retry recovers");
    assert_eq!(response.output, "2 + 3 = 5");
    let log = agent.take_effect_log().expect("recording");
    assert_eq!(
        log.records
            .iter()
            .map(|record| record.kind.family())
            .collect::<Vec<_>>(),
        [
            rig::effect::EffectFamily::Completion,
            rig::effect::EffectFamily::Completion,
            rig::effect::EffectFamily::Tool,
            rig::effect::EffectFamily::Completion,
        ],
        "the invalid call's turn, the retry, the real call, the answer"
    );
    crate::goldens::golden_effects("mock_invalid_tool_call_recovery", &log);
}
