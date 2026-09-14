//! Matrix H's reprompt cells: in `Tool` output mode the run validates the
//! answer itself — a text answer where the output tool's call was due, or
//! a call missing a required field, is reprompted once with feedback
//! (`RunSpec::DEFAULT_OUTPUT_RETRIES`). No live model in the corpus
//! produced either on request, so both are scripted from the mock model.

use rig::agent::AgentBuilder;
use rig::effect::EffectFamily;
use rig::run::OutputMode;
use rig::test_utils::{MockCompletionModel, MockTurn};
use serde_json::json;

use crate::goldens::{event_schema, families};

const PREAMBLE: &str = "You are a concise assistant. Answer directly.";
const PROMPT: &str = "Return a concise event object for a local Rust meetup in Seattle.";

fn event() -> serde_json::Value {
    json!({"title": "Seattle Rust Meetup", "category": "Technology", "summary": "A meetup."})
}

/// A text answer where the output tool's call was due: reprompted, then
/// the call settles the run.
#[tokio::test]
async fn output_tool_text_reprompt_effect_log_is_the_golden_fixture() {
    let agent = AgentBuilder::new(MockCompletionModel::from_turns([
        MockTurn::text("Seattle Rust Meetup, a technology meetup."),
        MockTurn::tool_call("call-1", "final_result", event()),
    ]))
    .name("golden")
    .preamble(PREAMBLE)
    .output_schema_raw(event_schema())
    .output_mode(OutputMode::Tool)
    .record_effects()
    .build();
    let response = agent
        .prompt(PROMPT)
        .max_turns(3)
        .await
        .expect("the reprompt recovers");
    assert_eq!(response.output, event().to_string());
    let log = agent.take_effect_log().expect("recording");
    assert_eq!(
        families(&log),
        [EffectFamily::Completion, EffectFamily::Completion]
    );
    crate::goldens::golden_effects("mock_output_tool_text_reprompt", &log);
}

/// An output-tool call missing a required field: reprompted with the
/// missing field named, then the complete call settles the run.
#[tokio::test]
async fn output_tool_missing_field_reprompt_effect_log_is_the_golden_fixture() {
    let agent = AgentBuilder::new(MockCompletionModel::from_turns([
        MockTurn::tool_call(
            "call-1",
            "final_result",
            json!({"title": "Seattle Rust Meetup", "category": "Technology"}),
        ),
        MockTurn::tool_call("call-2", "final_result", event()),
    ]))
    .name("golden")
    .preamble(PREAMBLE)
    .output_schema_raw(event_schema())
    .output_mode(OutputMode::Tool)
    .record_effects()
    .build();
    let response = agent
        .prompt(PROMPT)
        .max_turns(3)
        .await
        .expect("the reprompt recovers");
    assert_eq!(response.output, event().to_string());
    let log = agent.take_effect_log().expect("recording");
    assert_eq!(
        families(&log),
        [EffectFamily::Completion, EffectFamily::Completion]
    );
    crate::goldens::golden_effects("mock_output_tool_missing_field_reprompt", &log);
}
