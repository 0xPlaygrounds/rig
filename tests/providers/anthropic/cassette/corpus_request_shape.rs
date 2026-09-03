//! Matrix E of the effect corpus: the request-shape axes that change the
//! run spec, and so the header hash every golden carries and every
//! record's `CompletionRequest`. Producers of the goldens
//! `crates/rig-verify/tests/corpus_request_shape.rs` replays by both
//! interpreters; the matrix's enumeration lives there.
//!
//! Every cell is recorded once against the real Anthropic wire
//! (`CLAUDE_SONNET_4_6`, temperature 0) under
//! `tests/cassettes/anthropic/corpus_request_shape/`, and its golden is
//! generated from the replayed cassette under `RIG_REGENERATE_GOLDEN=1`.

use futures::StreamExt;
use rig::agent::MultiTurnStreamItem;
use rig::effect::EffectFamily;
use rig::message::{AssistantContent, ToolChoice};
use rig::prelude::*;
use rig::providers::anthropic::completion::CLAUDE_SONNET_4_6;

use super::super::support::with_anthropic_corpus_request_shape_cassette;
use crate::goldens::{EVENT_SCHEMA, event_schema, families, prior_history};
use crate::support::{
    Adder, BASIC_PREAMBLE, BASIC_PROMPT, CONTEXT_DOCS, CONTEXT_PROMPT, STRUCTURED_OUTPUT_PROMPT,
    TOOLS_PREAMBLE,
};

const ADD_PROMPT: &str = "Use the add tool to add 17 and 25, then reply with just the number.";
const NO_TOOL_PROMPT: &str = "What is 17 + 25? Reply with just the number.";
const NAME_PROMPT: &str = "What is my name? Reply with just the name.";
const THINKING_PROMPT: &str =
    "Think briefly, then answer: what is 12 * 12? Reply with just the number.";

/// Extended thinking with a budget: in `adaptive` mode Sonnet 4.6 chose
/// not to think about a one-line arithmetic prompt (first recording), so
/// the cells enable it outright.
fn thinking_params() -> serde_json::Value {
    serde_json::json!({ "thinking": { "type": "enabled", "budget_tokens": 1024 } })
}

fn reasoning_blocks(log: &rig::effect_log::EffectLog) -> usize {
    log.records
        .iter()
        .filter_map(|record| match record.outcome.as_ref() {
            Ok(rig::effect::Outcome::Completion(response)) => Some(response),
            _ => None,
        })
        .flat_map(|response| response.choice.iter())
        .filter(|content| matches!(content, AssistantContent::Reasoning(_)))
        .count()
}

/// The last completion's text, as the golden's oracle reads it.
fn last_text(log: &rig::effect_log::EffectLog) -> String {
    log.records
        .iter()
        .rev()
        .find_map(|record| match record.outcome.as_ref() {
            Ok(rig::effect::Outcome::Completion(response)) => Some(
                response
                    .choice
                    .iter()
                    .filter_map(|content| match content {
                        AssistantContent::Text(text) => Some(text.text.clone()),
                        _ => None,
                    })
                    .collect::<String>(),
            ),
            _ => None,
        })
        .expect("a completion")
}

// -- tool_choice ------------------------------------------------------------

/// `tool_choice(Auto)` with `add` advertised: the model calls it.
#[tokio::test]
async fn tool_choice_auto_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_request_shape_cassette(
        "corpus_request_shape/tool_choice_auto",
        |client| async move {
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .name("golden")
                .preamble(TOOLS_PREAMBLE)
                .temperature(0.0)
                .tool_choice(ToolChoice::Auto)
                .tool(Adder)
                .record_effects()
                .build();
            let response = agent
                .prompt(ADD_PROMPT)
                .max_turns(3)
                .await
                .expect("the agent answers");
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
            crate::goldens::golden_effects("anthropic_request_shape_tool_choice_auto", &log);
        },
    )
    .await;
}

/// `tool_choice(Required)`: every turn must be a tool call, and the run
/// spec's tool choice applies to every turn, so the run never reaches a
/// text answer: after `max_turns(2)` model calls it ends in `MaxTurnsError`
/// with `[Completion, Tool, Completion, Tool]` recorded. The corpus pins
/// that this is what the engine does with a per-run `Required`.
#[tokio::test]
async fn tool_choice_required_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_request_shape_cassette(
        "corpus_request_shape/tool_choice_required",
        |client| async move {
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .name("golden")
                .preamble(TOOLS_PREAMBLE)
                .temperature(0.0)
                .tool_choice(ToolChoice::Required)
                .tool(Adder)
                .record_effects()
                .build();
            let error = agent
                .prompt(ADD_PROMPT)
                .max_turns(2)
                .await
                .expect_err("a forced tool choice never yields a text answer");
            assert!(
                matches!(
                    error,
                    rig::completion::PromptError::MaxTurnsError { max_turns: 2, .. }
                ),
                "{error:?}"
            );
            let log = agent.take_effect_log().expect("recording");
            assert_eq!(
                families(&log),
                [
                    EffectFamily::Completion,
                    EffectFamily::Tool,
                    EffectFamily::Completion,
                    EffectFamily::Tool
                ]
            );
            crate::goldens::golden_effects("anthropic_request_shape_tool_choice_required", &log);
        },
    )
    .await;
}

/// `tool_choice(Specific(add))`: the named tool is forced on every turn,
/// so, like `Required`, the run ends in `MaxTurnsError` after two forced
/// calls.
#[tokio::test]
async fn tool_choice_specific_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_request_shape_cassette(
        "corpus_request_shape/tool_choice_specific",
        |client| async move {
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .name("golden")
                .preamble(TOOLS_PREAMBLE)
                .temperature(0.0)
                .tool_choice(ToolChoice::Specific {
                    function_names: vec!["add".to_owned()],
                })
                .tool(Adder)
                .record_effects()
                .build();
            let error = agent
                .prompt(ADD_PROMPT)
                .max_turns(2)
                .await
                .expect_err("a forced tool choice never yields a text answer");
            assert!(
                matches!(
                    error,
                    rig::completion::PromptError::MaxTurnsError { max_turns: 2, .. }
                ),
                "{error:?}"
            );
            let log = agent.take_effect_log().expect("recording");
            assert_eq!(
                families(&log),
                [
                    EffectFamily::Completion,
                    EffectFamily::Tool,
                    EffectFamily::Completion,
                    EffectFamily::Tool
                ]
            );
            crate::goldens::golden_effects("anthropic_request_shape_tool_choice_specific", &log);
        },
    )
    .await;
}

/// `tool_choice(None)` with `add` advertised: no tool record exists. What
/// the wire did: Sonnet 4.6 answered `tool_choice: none` with an empty
/// `content: []` and `end_turn` under the tools preamble *and* under the
/// basic preamble (two recordings, then stop), so the record is a
/// completion whose choice is empty and the run's output is the empty
/// string. The cell pins that the engine carries an empty answer through
/// as an answer, and that the request holds `tool_choice: none` with the
/// tool still advertised.
#[tokio::test]
async fn tool_choice_none_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_request_shape_cassette(
        "corpus_request_shape/tool_choice_none",
        |client| async move {
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .name("golden")
                .preamble(BASIC_PREAMBLE)
                .temperature(0.0)
                .tool_choice(ToolChoice::None)
                .tool(Adder)
                .record_effects()
                .build();
            let response = agent
                .prompt(NO_TOOL_PROMPT)
                .max_turns(3)
                .await
                .expect("the agent answers");
            let log = agent.take_effect_log().expect("recording");
            assert_eq!(families(&log), [EffectFamily::Completion]);
            let request = match &log.records[0].kind {
                rig::effect::EffectKind::Completion { request, .. } => request,
                other => panic!("a completion, not {other:?}"),
            };
            assert_eq!(request.tool_choice, Some(ToolChoice::None));
            assert_eq!(request.tools.len(), 1, "add is still advertised");
            assert_eq!(
                response.output, "",
                "Sonnet 4.6 answers `tool_choice: none` with empty content; if this changes, the cell changes"
            );
            crate::goldens::golden_effects("anthropic_request_shape_tool_choice_none", &log);
        },
    )
    .await;
}

// -- sampling and params ---------------------------------------------------

/// `max_tokens(32)`: the request carries the cap; the answer stops at it.
#[tokio::test]
async fn max_tokens_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_request_shape_cassette(
        "corpus_request_shape/max_tokens",
        |client| async move {
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .name("golden")
                .preamble(BASIC_PREAMBLE)
                .temperature(0.0)
                .max_tokens(32)
                .record_effects()
                .build();
            let response = agent.prompt(BASIC_PROMPT).await.expect("the agent answers");
            assert!(!response.output.is_empty());
            let log = agent.take_effect_log().expect("recording");
            assert_eq!(families(&log), [EffectFamily::Completion]);
            let request = match &log.records[0].kind {
                rig::effect::EffectKind::Completion { request, .. } => request,
                other => panic!("a completion, not {other:?}"),
            };
            assert_eq!(request.max_tokens, Some(32));
            crate::goldens::golden_effects("anthropic_request_shape_max_tokens", &log);
        },
    )
    .await;
}

/// `additional_params(thinking: adaptive)`, unary: the record's completion
/// carries a reasoning block with its signature.
#[tokio::test]
async fn thinking_unary_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_request_shape_cassette(
        "corpus_request_shape/thinking_unary",
        |client| async move {
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .name("golden")
                .preamble(BASIC_PREAMBLE)
                .additional_params(thinking_params())
                .record_effects()
                .build();
            let response = agent
                .prompt(THINKING_PROMPT)
                .await
                .expect("the agent answers");
            assert!(response.output.contains("144"), "{}", response.output);
            let log = agent.take_effect_log().expect("recording");
            assert_eq!(families(&log), [EffectFamily::Completion]);
            assert!(reasoning_blocks(&log) >= 1, "the completion reasons");
            crate::goldens::golden_effects("anthropic_request_shape_thinking_unary", &log);
        },
    )
    .await;
}

/// The same, streamed with its events kept: the reasoning deltas and the
/// block's signature are on the record, and both interpreters carry them.
#[tokio::test]
async fn thinking_streamed_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_request_shape_cassette(
        "corpus_request_shape/thinking_streamed",
        |client| async move {
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .name("golden")
                .preamble(BASIC_PREAMBLE)
                .additional_params(thinking_params())
                .record_effects_with_events()
                .build();
            let mut stream = agent.stream_prompt(THINKING_PROMPT).stream().await;
            let mut output = None;
            while let Some(item) = stream.next().await {
                if let MultiTurnStreamItem::FinalResponse(response) =
                    item.expect("the stream yields")
                {
                    output = Some(response.output);
                }
            }
            drop(stream);
            let output = output.expect("a final response");
            assert!(output.contains("144"), "{output}");
            let log = agent.take_effect_log().expect("recording");
            assert_eq!(families(&log), [EffectFamily::Completion]);
            assert!(log.records[0].events.is_some(), "events are kept");
            assert!(reasoning_blocks(&log) >= 1, "the completion reasons");
            crate::goldens::golden_effects("anthropic_request_shape_thinking_streamed", &log);
        },
    )
    .await;
}

// -- preamble and context ---------------------------------------------------

/// Two static `context` documents: the request holds them and the answer
/// uses them.
#[tokio::test]
async fn static_context_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_request_shape_cassette(
        "corpus_request_shape/static_context",
        |client| async move {
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .name("golden")
                .preamble(BASIC_PREAMBLE)
                .temperature(0.0)
                .context(CONTEXT_DOCS[0])
                .context(CONTEXT_DOCS[1])
                .record_effects()
                .build();
            let response = agent
                .prompt(CONTEXT_PROMPT)
                .await
                .expect("the agent answers");
            assert!(!response.output.is_empty());
            let log = agent.take_effect_log().expect("recording");
            assert_eq!(families(&log), [EffectFamily::Completion]);
            let request = match &log.records[0].kind {
                rig::effect::EffectKind::Completion { request, .. } => request,
                other => panic!("a completion, not {other:?}"),
            };
            assert_eq!(request.documents.len(), 2, "{:?}", request.documents);
            crate::goldens::golden_effects("anthropic_request_shape_static_context", &log);
        },
    )
    .await;
}

/// `append_preamble`: the spec's preamble is the base and the document.
#[tokio::test]
async fn append_preamble_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_request_shape_cassette(
        "corpus_request_shape/append_preamble",
        |client| async move {
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .name("golden")
                .preamble(BASIC_PREAMBLE)
                .append_preamble("Always end your answer with the word DONE.")
                .temperature(0.0)
                .record_effects()
                .build();
            let response = agent.prompt(BASIC_PROMPT).await.expect("the agent answers");
            assert!(response.output.contains("DONE"), "{}", response.output);
            let log = agent.take_effect_log().expect("recording");
            assert_eq!(families(&log), [EffectFamily::Completion]);
            crate::goldens::golden_effects("anthropic_request_shape_append_preamble", &log);
        },
    )
    .await;
}

/// `without_preamble`: the request carries no system prompt at all.
#[tokio::test]
async fn without_preamble_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_request_shape_cassette(
        "corpus_request_shape/without_preamble",
        |client| async move {
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .name("golden")
                .without_preamble()
                .temperature(0.0)
                .record_effects()
                .build();
            let response = agent.prompt(BASIC_PROMPT).await.expect("the agent answers");
            assert!(!response.output.is_empty());
            let log = agent.take_effect_log().expect("recording");
            assert_eq!(families(&log), [EffectFamily::Completion]);
            let request = match &log.records[0].kind {
                rig::effect::EffectKind::Completion { request, .. } => request,
                other => panic!("a completion, not {other:?}"),
            };
            assert_eq!(request.system_instructions(), None);
            crate::goldens::golden_effects("anthropic_request_shape_without_preamble", &log);
        },
    )
    .await;
}

// -- output and history -----------------------------------------------------

/// `output_schema_raw`, unary: the answer is the schema's object.
#[tokio::test]
async fn output_schema_unary_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_request_shape_cassette(
        "corpus_request_shape/output_schema_unary",
        |client| async move {
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .name("golden")
                .preamble(BASIC_PREAMBLE)
                .temperature(0.0)
                .output_schema_raw(event_schema())
                .record_effects()
                .build();
            let response = agent
                .prompt(STRUCTURED_OUTPUT_PROMPT)
                .await
                .expect("the agent answers");
            let object: serde_json::Value =
                serde_json::from_str(&response.output).expect("the answer is the schema's object");
            assert!(object["title"].is_string(), "{object}");
            let log = agent.take_effect_log().expect("recording");
            assert_eq!(families(&log), [EffectFamily::Completion]);
            assert_eq!(last_text(&log), response.output);
            crate::goldens::golden_effects("anthropic_request_shape_output_schema_unary", &log);
        },
    )
    .await;
}

/// `output_schema_raw`, streamed with events kept.
#[tokio::test]
async fn output_schema_streamed_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_request_shape_cassette(
        "corpus_request_shape/output_schema_streamed",
        |client| async move {
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .name("golden")
                .preamble(BASIC_PREAMBLE)
                .temperature(0.0)
                .output_schema_raw(event_schema())
                .record_effects_with_events()
                .build();
            let mut stream = agent.stream_prompt(STRUCTURED_OUTPUT_PROMPT).stream().await;
            let mut output = None;
            while let Some(item) = stream.next().await {
                if let MultiTurnStreamItem::FinalResponse(response) =
                    item.expect("the stream yields")
                {
                    output = Some(response.output);
                }
            }
            drop(stream);
            let output = output.expect("a final response");
            let object: serde_json::Value =
                serde_json::from_str(&output).expect("the answer is the schema's object");
            assert!(object["title"].is_string(), "{object}");
            let log = agent.take_effect_log().expect("recording");
            assert_eq!(families(&log), [EffectFamily::Completion]);
            assert!(log.records[0].events.is_some(), "events are kept");
            crate::goldens::golden_effects("anthropic_request_shape_output_schema_streamed", &log);
        },
    )
    .await;
}

/// A prior history on the runner: the first record's request already
/// holds two turns before the prompt.
#[tokio::test]
async fn prior_history_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_request_shape_cassette(
        "corpus_request_shape/prior_history",
        |client| async move {
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .name("golden")
                .preamble(BASIC_PREAMBLE)
                .temperature(0.0)
                .record_effects()
                .build();
            let response = agent
                .prompt(NAME_PROMPT)
                .history(prior_history())
                .await
                .expect("the agent answers");
            assert!(response.output.contains("Ada"), "{}", response.output);
            let log = agent.take_effect_log().expect("recording");
            assert_eq!(families(&log), [EffectFamily::Completion]);
            let request = match &log.records[0].kind {
                rig::effect::EffectKind::Completion { request, .. } => request,
                other => panic!("a completion, not {other:?}"),
            };
            let turns = request
                .chat_history
                .iter()
                .filter(|message| !matches!(message, rig::message::Message::System { .. }))
                .count();
            assert_eq!(
                turns, 3,
                "two prior turns and the prompt: {:?}",
                request.chat_history
            );
            crate::goldens::golden_effects("anthropic_request_shape_prior_history", &log);
        },
    )
    .await;
}

/// The schema literal both the producer and the replay build the program
/// from is one string; a drift between them is a spec-hash refusal.
#[test]
fn the_event_schema_is_the_literal() {
    assert_eq!(
        serde_json::to_value(event_schema()).expect("a schema serializes"),
        serde_json::from_str::<serde_json::Value>(EVENT_SCHEMA).expect("the literal parses")
    );
}
