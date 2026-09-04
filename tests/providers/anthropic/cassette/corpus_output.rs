//! Matrix H of the effect corpus: output modes, on the Anthropic wire
//! (`CLAUDE_SONNET_4_6`, temperature 0). The event schema of Matrix E
//! under `Tool` and `Prompted`, beside a real tool, under each tool
//! choice, and under extended thinking. Producers of the goldens
//! `crates/rig-verify/tests/corpus_output.rs` replays by both
//! interpreters; the enumeration lives there. Every cell is a new
//! recording under `tests/cassettes/anthropic/corpus_output/`.

use futures::StreamExt;
use rig::agent::MultiTurnStreamItem;
use rig::effect::{EffectFamily, EffectKind};
use rig::message::ToolChoice;
use rig::prelude::*;
use rig::providers::anthropic::completion::CLAUDE_SONNET_4_6;
use rig::run::OutputMode;

use super::super::support::with_anthropic_corpus_output_cassette;
use crate::goldens::{event_schema, families};
use crate::support::{Adder, BASIC_PREAMBLE, STRUCTURED_OUTPUT_PROMPT, TOOLS_PREAMBLE};

const SUM_EVENT_PROMPT: &str = "Use the add tool to add 17 and 25, then return a concise event object for a Rust meetup in Seattle whose summary states the sum.";

fn request_at(log: &rig::effect_log::EffectLog, at: usize) -> &rig::completion::CompletionRequest {
    match &log.records[at].kind {
        EffectKind::Completion { request, .. } => request,
        other => panic!("record {at} is a completion, not {other:?}"),
    }
}

fn assert_event(output: &str) {
    let object: serde_json::Value =
        serde_json::from_str(output).unwrap_or_else(|_| panic!("the schema's object: {output}"));
    assert!(
        object["title"].is_string() && object["summary"].is_string(),
        "{object}"
    );
}

fn tool_names(request: &rig::completion::CompletionRequest) -> Vec<&str> {
    request
        .tools
        .iter()
        .map(|tool| tool.name.as_str())
        .collect()
}

async fn final_output(stream: &mut rig::agent::StreamingResult) -> String {
    let mut output = None;
    while let Some(item) = stream.next().await {
        if let MultiTurnStreamItem::FinalResponse(response) = item.expect("the stream yields") {
            output = Some(response.output);
        }
    }
    output.expect("a final response")
}

/// `Tool` mode: the request advertises the synthetic `final_result`
/// tool and augments the preamble; the model's call to it is the answer,
/// settled by the run without a dispatch.
#[tokio::test]
async fn tool_unary_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_output_cassette("corpus_output/tool_unary", |client| async move {
        let agent = client
            .agent(CLAUDE_SONNET_4_6)
            .name("golden")
            .preamble(BASIC_PREAMBLE)
            .temperature(0.0)
            .output_schema_raw(event_schema())
            .output_mode(OutputMode::Tool)
            .record_effects()
            .build();
        let response = agent
            .prompt(STRUCTURED_OUTPUT_PROMPT)
            .await
            .expect("the agent answers");
        assert_event(&response.output);
        let log = agent.take_effect_log().expect("recording");
        assert_eq!(families(&log), [EffectFamily::Completion]);
        let request = request_at(&log, 0);
        assert_eq!(tool_names(request), ["final_result"]);
        assert!(
            request
                .system_instructions()
                .is_some_and(|system| system.contains("`final_result`")),
            "the preamble is augmented"
        );
        assert!(
            request.output_schema.is_none(),
            "no native schema in Tool mode"
        );
        crate::goldens::golden_effects("anthropic_output_tool_unary", &log);
    })
    .await;
}

/// The same, streamed with events.
#[tokio::test]
async fn tool_streamed_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_output_cassette("corpus_output/tool_streamed", |client| async move {
        let agent = client
            .agent(CLAUDE_SONNET_4_6)
            .name("golden")
            .preamble(BASIC_PREAMBLE)
            .temperature(0.0)
            .output_schema_raw(event_schema())
            .output_mode(OutputMode::Tool)
            .record_effects_with_events()
            .build();
        let mut stream = agent.stream_prompt(STRUCTURED_OUTPUT_PROMPT).stream().await;
        let output = final_output(&mut stream).await;
        drop(stream);
        assert_event(&output);
        let log = agent.take_effect_log().expect("recording");
        assert_eq!(families(&log), [EffectFamily::Completion]);
        assert!(log.records[0].events.is_some(), "events are kept");
        assert_eq!(tool_names(request_at(&log, 0)), ["final_result"]);
        crate::goldens::golden_effects("anthropic_output_tool_streamed", &log);
    })
    .await;
}

/// `Prompted` mode: no tool, no native schema; the preamble carries the
/// schema and the answer is JSON text.
#[tokio::test]
async fn prompted_unary_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_output_cassette("corpus_output/prompted_unary", |client| async move {
        let agent = client
            .agent(CLAUDE_SONNET_4_6)
            .name("golden")
            .preamble(BASIC_PREAMBLE)
            .temperature(0.0)
            .output_schema_raw(event_schema())
            .output_mode(OutputMode::Prompted)
            .record_effects()
            .build();
        let response = agent
            .prompt(STRUCTURED_OUTPUT_PROMPT)
            .await
            .expect("the agent answers");
        assert_event(&response.output);
        let log = agent.take_effect_log().expect("recording");
        assert_eq!(families(&log), [EffectFamily::Completion]);
        let request = request_at(&log, 0);
        assert!(request.tools.is_empty());
        assert!(request.output_schema.is_none());
        assert!(
            request
                .system_instructions()
                .is_some_and(|system| system.contains("JSON Schema")),
            "the preamble carries the schema"
        );
        crate::goldens::golden_effects("anthropic_output_prompted_unary", &log);
    })
    .await;
}

/// The same, streamed with events.
#[tokio::test]
async fn prompted_streamed_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_output_cassette("corpus_output/prompted_streamed", |client| async move {
        let agent = client
            .agent(CLAUDE_SONNET_4_6)
            .name("golden")
            .preamble(BASIC_PREAMBLE)
            .temperature(0.0)
            .output_schema_raw(event_schema())
            .output_mode(OutputMode::Prompted)
            .record_effects_with_events()
            .build();
        let mut stream = agent.stream_prompt(STRUCTURED_OUTPUT_PROMPT).stream().await;
        let output = final_output(&mut stream).await;
        drop(stream);
        assert_event(&output);
        let log = agent.take_effect_log().expect("recording");
        assert_eq!(families(&log), [EffectFamily::Completion]);
        assert!(log.records[0].events.is_some(), "events are kept");
        crate::goldens::golden_effects("anthropic_output_prompted_streamed", &log);
    })
    .await;
}

/// `Tool` mode beside a real tool: the model calls `add` first (a
/// dispatch), then `final_result` (settled, no dispatch).
#[tokio::test]
async fn tool_with_real_tool_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_output_cassette(
        "corpus_output/tool_with_real_tool",
        |client| async move {
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .name("golden")
                .preamble(TOOLS_PREAMBLE)
                .temperature(0.0)
                .output_schema_raw(event_schema())
                .output_mode(OutputMode::Tool)
                .tool(Adder)
                .record_effects()
                .build();
            let response = agent
                .prompt(SUM_EVENT_PROMPT)
                .max_turns(3)
                .await
                .expect("the agent answers");
            assert_event(&response.output);
            assert!(response.output.contains("42"), "{}", response.output);
            let log = agent.take_effect_log().expect("recording");
            assert_eq!(
                families(&log),
                [
                    EffectFamily::Completion,
                    EffectFamily::Tool,
                    EffectFamily::Completion
                ]
            );
            assert_eq!(tool_names(request_at(&log, 0)), ["add", "final_result"]);
            crate::goldens::golden_effects("anthropic_output_tool_with_real_tool", &log);
        },
    )
    .await;
}

/// `Prompted` mode beside a real tool.
#[tokio::test]
async fn prompted_with_real_tool_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_output_cassette(
        "corpus_output/prompted_with_real_tool",
        |client| async move {
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .name("golden")
                .preamble(TOOLS_PREAMBLE)
                .temperature(0.0)
                .output_schema_raw(event_schema())
                .output_mode(OutputMode::Prompted)
                .tool(Adder)
                .record_effects()
                .build();
            let response = agent
                .prompt(SUM_EVENT_PROMPT)
                .max_turns(3)
                .await
                .expect("the agent answers");
            assert_event(&response.output);
            let log = agent.take_effect_log().expect("recording");
            assert_eq!(
                families(&log),
                [
                    EffectFamily::Completion,
                    EffectFamily::Tool,
                    EffectFamily::Completion
                ]
            );
            assert_eq!(tool_names(request_at(&log, 0)), ["add"]);
            crate::goldens::golden_effects("anthropic_output_prompted_with_real_tool", &log);
        },
    )
    .await;
}

/// `Tool` mode with `tool_choice: Specific(final_result)`: the output
/// tool is the only call allowed, and it is called.
#[tokio::test]
async fn tool_choice_specific_output_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_output_cassette(
        "corpus_output/tool_choice_specific_output",
        |client| async move {
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .name("golden")
                .preamble(BASIC_PREAMBLE)
                .temperature(0.0)
                .output_schema_raw(event_schema())
                .output_mode(OutputMode::Tool)
                .tool_choice(ToolChoice::Specific {
                    function_names: vec!["final_result".to_owned()],
                })
                .record_effects()
                .build();
            let response = agent
                .prompt(STRUCTURED_OUTPUT_PROMPT)
                .await
                .expect("the agent answers");
            assert_event(&response.output);
            let log = agent.take_effect_log().expect("recording");
            assert_eq!(families(&log), [EffectFamily::Completion]);
            assert_eq!(tool_names(request_at(&log, 0)), ["final_result"]);
            crate::goldens::golden_effects("anthropic_output_tool_choice_specific_output", &log);
        },
    )
    .await;
}

/// `Tool` mode with `tool_choice: Required`: the forced call is the
/// output tool's, which settles the run before the choice can force a
/// second turn.
#[tokio::test]
async fn tool_choice_required_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_output_cassette(
        "corpus_output/tool_choice_required",
        |client| async move {
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .name("golden")
                .preamble(BASIC_PREAMBLE)
                .temperature(0.0)
                .output_schema_raw(event_schema())
                .output_mode(OutputMode::Tool)
                .tool_choice(ToolChoice::Required)
                .record_effects()
                .build();
            let response = agent
                .prompt(STRUCTURED_OUTPUT_PROMPT)
                .max_turns(2)
                .await
                .expect("the agent answers");
            assert_event(&response.output);
            let log = agent.take_effect_log().expect("recording");
            assert_eq!(families(&log), [EffectFamily::Completion]);
            crate::goldens::golden_effects("anthropic_output_tool_choice_required", &log);
        },
    )
    .await;
}

/// `Tool` mode with `tool_choice: None`: the output tool cannot be
/// called, so the mode resolves to `Native` — a native schema, no tools —
/// rather than a turn that cannot finalize.
#[tokio::test]
async fn tool_under_none_degrades_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_output_cassette(
        "corpus_output/tool_under_none_degrades",
        |client| async move {
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .name("golden")
                .preamble(BASIC_PREAMBLE)
                .temperature(0.0)
                .output_schema_raw(event_schema())
                .output_mode(OutputMode::Tool)
                .tool_choice(ToolChoice::None)
                .record_effects()
                .build();
            let response = agent
                .prompt(STRUCTURED_OUTPUT_PROMPT)
                .await
                .expect("the agent answers");
            assert_event(&response.output);
            let log = agent.take_effect_log().expect("recording");
            assert_eq!(families(&log), [EffectFamily::Completion]);
            let request = request_at(&log, 0);
            assert!(
                request.tools.is_empty(),
                "no output tool under tool_choice none"
            );
            assert!(request.output_schema.is_some(), "the native schema instead");
            crate::goldens::golden_effects("anthropic_output_tool_under_none_degrades", &log);
        },
    )
    .await;
}

/// `Tool` mode under extended thinking: the record holds a reasoning
/// block and the output tool's call.
#[tokio::test]
async fn tool_thinking_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_output_cassette("corpus_output/tool_thinking", |client| async move {
        let agent = client
            .agent(CLAUDE_SONNET_4_6)
            .name("golden")
            .preamble(BASIC_PREAMBLE)
            .additional_params(
                serde_json::json!({ "thinking": { "type": "enabled", "budget_tokens": 1024 } }),
            )
            .output_schema_raw(event_schema())
            .output_mode(OutputMode::Tool)
            .record_effects()
            .build();
        let response = agent
            .prompt(STRUCTURED_OUTPUT_PROMPT)
            .await
            .expect("the agent answers");
        assert_event(&response.output);
        let log = agent.take_effect_log().expect("recording");
        assert_eq!(families(&log), [EffectFamily::Completion]);
        crate::goldens::golden_effects("anthropic_output_tool_thinking", &log);
    })
    .await;
}
