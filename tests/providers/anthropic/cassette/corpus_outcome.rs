//! Matrix D of the effect corpus: cancellation and failure outcomes, and
//! the run beyond a single answer. Producers of the goldens
//! `crates/rig-verify/tests/corpus_outcome.rs` replays by both
//! interpreters; the enumeration lives there.
//!
//! The cancel and tool-error cells are new recordings under
//! `tests/cassettes/anthropic/corpus_outcome/`; the model-error cells
//! record the wire's own 401 under an invalid key; the turn-budget cells
//! reuse the tool-call-turn cassette, since a budget changes when the run
//! stops, not what it asks.

use futures::StreamExt;
use rig::agent::MultiTurnStreamItem;
use rig::completion::PromptError;
use rig::effect::{EffectFamily, Outcome};
use rig::error::ErrorKind;
use rig::prelude::*;
use rig::providers::anthropic::completion::CLAUDE_SONNET_4_6;
use rig::streaming::{Delta, StreamEvent};

use super::super::support::{
    with_anthropic_cassette, with_anthropic_cassette_bogus_key,
    with_anthropic_corpus_outcome_cassette,
};
use crate::goldens::{BROKEN_ADD, FailingAdd, WriteNote, families};
use crate::support::{Adder, BASIC_PREAMBLE, BASIC_PROMPT, TOOLS_PREAMBLE};

const ADD_PROMPT: &str = "Use the add tool to add 17 and 25, then reply with just the number.";
const NOTE_PREAMBLE: &str =
    "You are a note-taking assistant. Use the write_note tool to save notes.";
const NOTE_PROMPT: &str = "Save a note titled 'Rust' whose body is a 400-word essay on the history of the Rust programming language, then reply with just the word saved.";

async fn final_output(stream: &mut rig::agent::StreamingResult) -> String {
    let mut output = None;
    while let Some(item) = stream.next().await {
        if let MultiTurnStreamItem::FinalResponse(response) = item.expect("the stream yields") {
            output = Some(response.output);
        }
    }
    output.expect("a final response")
}

fn tool_outcome(log: &rig::effect_log::EffectLog) -> &rig::tool::ToolResult {
    log.records
        .iter()
        .find_map(|record| match &record.outcome {
            Ok(Outcome::ToolResult { result, .. }) => Some(result),
            _ => None,
        })
        .expect("a tool record")
}

/// A streamed tool-call turn whose consumer drops the stream at the first
/// tool-call delta: the completion is recorded as `Cancelled` and the tool
/// never runs. The call's arguments are a long note (the first recording,
/// an `add` with two integers, had streamed to its end before the drop
/// landed and recorded a success), so the drop lands mid-call.
#[tokio::test]
async fn cancel_after_tool_call_delta_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_outcome_cassette(
        "corpus_outcome/cancel_after_tool_call_delta",
        |client| async move {
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .name("golden")
                .preamble(NOTE_PREAMBLE)
                .temperature(0.0)
                .tool(WriteNote)
                .record_effects_with_events()
                .build();
            {
                let mut stream = agent.stream_prompt(NOTE_PROMPT).max_turns(3).stream().await;
                while let Some(item) = stream.next().await {
                    if let Ok(MultiTurnStreamItem::StreamAssistantItem(StreamEvent::BlockDelta {
                        delta: Delta::ToolName { .. } | Delta::ToolArguments { .. },
                        ..
                    })) = item
                    {
                        break;
                    }
                }
                // Dropped here, mid-call.
            }
            for _ in 0..64 {
                tokio::task::yield_now().await;
            }
            let log = agent.take_effect_log().expect("recording");
            assert_eq!(families(&log), [EffectFamily::Completion]);
            let report = log.records[0]
                .outcome
                .as_ref()
                .expect_err("a dropped stream is recorded as a cancel");
            assert_eq!(report.kind, ErrorKind::Cancelled, "{report:?}");
            crate::goldens::golden_effects("anthropic_outcome_cancel_after_tool_call_delta", &log);
        },
    )
    .await;
}

/// A tool that fails: the tool record's outcome is a failed result, the
/// model sees the failure and answers around it.
#[tokio::test]
async fn tool_error_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_outcome_cassette("corpus_outcome/tool_error", |client| async move {
        let agent = client
            .agent(CLAUDE_SONNET_4_6)
            .name("golden")
            .preamble(TOOLS_PREAMBLE)
            .temperature(0.0)
            .tool(FailingAdd)
            .record_effects()
            .build();
        let response = agent
            .prompt(ADD_PROMPT)
            .max_turns(3)
            .await
            .expect("the agent answers around the failure");
        assert!(!response.output.is_empty());
        let log = agent.take_effect_log().expect("recording");
        assert_eq!(
            families(&log),
            [
                EffectFamily::Completion,
                EffectFamily::Tool,
                EffectFamily::Completion
            ]
        );
        let result = tool_outcome(&log);
        assert!(result.is_error(), "{result:?}");
        assert!(result.output().render().contains(BROKEN_ADD), "{result:?}");
        crate::goldens::golden_effects("anthropic_outcome_tool_error", &log);
    })
    .await;
}

/// The same, streamed with events kept.
#[tokio::test]
async fn tool_error_streamed_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_outcome_cassette(
        "corpus_outcome/tool_error_streamed",
        |client| async move {
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .name("golden")
                .preamble(TOOLS_PREAMBLE)
                .temperature(0.0)
                .tool(FailingAdd)
                .record_effects_with_events()
                .build();
            let mut stream = agent.stream_prompt(ADD_PROMPT).max_turns(3).stream().await;
            let output = final_output(&mut stream).await;
            drop(stream);
            assert!(!output.is_empty());
            let log = agent.take_effect_log().expect("recording");
            assert_eq!(
                families(&log),
                [
                    EffectFamily::Completion,
                    EffectFamily::Tool,
                    EffectFamily::Completion
                ]
            );
            assert!(log.records[0].events.is_some(), "events are kept");
            assert!(tool_outcome(&log).is_error());
            crate::goldens::golden_effects("anthropic_outcome_tool_error_streamed", &log);
        },
    )
    .await;
}

/// The wire's own error: an invalid key, a 401 envelope. The completion
/// record's outcome is the provider's error and the run fails at it.
#[tokio::test]
async fn model_error_effect_log_is_the_golden_fixture() {
    with_anthropic_cassette_bogus_key("corpus_outcome/model_error", |client| async move {
        let agent = client
            .agent(CLAUDE_SONNET_4_6)
            .name("golden")
            .preamble(BASIC_PREAMBLE)
            .temperature(0.0)
            .record_effects()
            .build();
        let error = agent
            .prompt(BASIC_PROMPT)
            .await
            .expect_err("an invalid key is refused");
        let kind = match &error {
            PromptError::Report(report) => report.kind,
            other => panic!("a report, not {other:?}"),
        };
        assert_eq!(kind, ErrorKind::ProviderResponse, "{error:?}");
        let log = agent.take_effect_log().expect("recording");
        assert_eq!(families(&log), [EffectFamily::Completion]);
        let report = log.records[0]
            .outcome
            .as_ref()
            .expect_err("the record holds the provider's error");
        assert_eq!(report.kind, ErrorKind::ProviderResponse);
        assert_eq!(report.http_status, Some(401), "{report:?}");
        crate::goldens::golden_effects("anthropic_outcome_model_error", &log);
    })
    .await;
}

/// The same, streamed: the error arrives as the stream's first item.
#[tokio::test]
async fn model_error_streamed_effect_log_is_the_golden_fixture() {
    with_anthropic_cassette_bogus_key("corpus_outcome/model_error_streamed", |client| async move {
        let agent = client
            .agent(CLAUDE_SONNET_4_6)
            .name("golden")
            .preamble(BASIC_PREAMBLE)
            .temperature(0.0)
            .record_effects_with_events()
            .build();
        let mut stream = agent.stream_prompt(BASIC_PROMPT).stream().await;
        let mut kinds = Vec::new();
        while let Some(item) = stream.next().await {
            if let Err(error) = item {
                kinds.push(error.to_string());
            }
        }
        drop(stream);
        assert_eq!(kinds.len(), 1, "one error item: {kinds:?}");
        let log = agent.take_effect_log().expect("recording");
        assert_eq!(families(&log), [EffectFamily::Completion]);
        let report = log.records[0]
            .outcome
            .as_ref()
            .expect_err("the record holds the provider's error");
        assert_eq!(report.kind, ErrorKind::ProviderResponse, "{report:?}");
        crate::goldens::golden_effects("anthropic_outcome_model_error_streamed", &log);
    })
    .await;
}

/// The runner's budget exhausted with a tool call pending: one model call
/// allowed, the tool runs, the next call is refused by the budget. Two
/// records, then `MaxTurnsError`. Its own recording: the run makes one
/// request, and a cassette with a second interaction refuses to leave it
/// unused.
#[tokio::test]
async fn max_turns_exhausted_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_outcome_cassette(
        "corpus_outcome/max_turns_exhausted",
        |client| async move {
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .name("golden")
                .preamble(TOOLS_PREAMBLE)
                .temperature(0.0)
                .tool(Adder)
                .record_effects()
                .build();
            let error = agent
                .prompt(ADD_PROMPT)
                .max_turns(1)
                .await
                .expect_err("one call cannot finish a tool turn");
            assert!(
                matches!(error, PromptError::MaxTurnsError { max_turns: 1, .. }),
                "{error:?}"
            );
            let log = agent.take_effect_log().expect("recording");
            assert_eq!(
                families(&log),
                [EffectFamily::Completion, EffectFamily::Tool]
            );
            crate::goldens::golden_effects("anthropic_outcome_max_turns_exhausted", &log);
        },
    )
    .await;
}

/// The builder's `default_max_turns` is in the spec the header hashes; the
/// runner's `max_turns` is not. This cell is the tool-call turn under a
/// default budget of three and no runner budget: its records are the
/// `anthropic_tool_call_turn` golden's, its header is another program's.
#[tokio::test]
async fn default_max_turns_effect_log_is_the_golden_fixture() {
    with_anthropic_cassette("effect_corpus/tool_call_turn", |client| async move {
        let agent = client
            .agent(CLAUDE_SONNET_4_6)
            .name("golden")
            .preamble(TOOLS_PREAMBLE)
            .temperature(0.0)
            .default_max_turns(3)
            .tool(Adder)
            .record_effects()
            .build();
        let response = agent.prompt(ADD_PROMPT).await.expect("the agent answers");
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
        crate::goldens::golden_effects("anthropic_outcome_default_max_turns", &log);
    })
    .await;
}
