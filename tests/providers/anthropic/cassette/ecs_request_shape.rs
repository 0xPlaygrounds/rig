//! Provider-executed native counterparts of the request-shape golden corpus.
//! Original inputs/helpers are shared; every run uses native ECS systems.

use crate::ecs_agent::EcsAgent;
use bevy_ecs::prelude::*;
use rig_ecs::{agent::*, systems::spawn_run};

use rig::effect::EffectFamily;
use rig::message::ToolChoice;
use rig::prelude::*;
use rig::providers::anthropic::completion::CLAUDE_SONNET_4_6;

use super::super::support::with_anthropic_corpus_request_shape_cassette;
use crate::goldens::{EVENT_SCHEMA, families, prior_history};
use crate::support::{
    Adder, BASIC_PREAMBLE, BASIC_PROMPT, CONTEXT_DOCS, CONTEXT_PROMPT, STRUCTURED_OUTPUT_PROMPT,
    TOOLS_PREAMBLE,
};

use super::corpus_request_shape::{
    ADD_PROMPT, NAME_PROMPT, NO_TOOL_PROMPT, THINKING_PROMPT, last_text, reasoning_blocks,
    thinking_params,
};

// -- tool_choice ------------------------------------------------------------

/// `tool_choice(Auto)` with `add` advertised: the model calls it.
#[tokio::test]
async fn tool_choice_auto_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_request_shape_cassette(
        "corpus_request_shape/tool_choice_auto",
        |client| async move {
            let mut ecs = EcsAgent::for_golden(
                client.completion_model(CLAUDE_SONNET_4_6),
                TOOLS_PREAMBLE,
                false,
            );
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert(Temperature(Some(0.0)));
            ecs.tool(Adder);
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert(ToolChoiceSpec(Some(ToolChoice::Auto)));
            let history = vec![];
            let run = spawn_run(
                ecs.app.world_mut(),
                ecs.agent,
                &history,
                ADD_PROMPT,
                false,
                Some(3),
            );
            let output = ecs.wait_for_success(run).await;
            assert!(output.contains("42"), "{}", output);
            let log = ecs.effect_log();
            assert_eq!(
                families(&log),
                [
                    EffectFamily::Completion,
                    EffectFamily::Tool,
                    EffectFamily::Completion
                ]
            );
            crate::ecs_goldens::golden_effects("anthropic_request_shape_tool_choice_auto", &log);
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
            let mut ecs = EcsAgent::for_golden(
                client.completion_model(CLAUDE_SONNET_4_6),
                TOOLS_PREAMBLE,
                false,
            );
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert(Temperature(Some(0.0)));
            ecs.tool(Adder);
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert(ToolChoiceSpec(Some(ToolChoice::Required)));
            let history = vec![];
            let run = spawn_run(
                ecs.app.world_mut(),
                ecs.agent,
                &history,
                ADD_PROMPT,
                false,
                Some(2),
            );
            let error = ecs
                .wait_for_outcome(run)
                .await
                .expect_err("forced tool choice exhausts budget");
            assert!(matches!(error, Failure::MaxTurns { limit: 2 }), "{error:?}");
            let log = ecs.effect_log();
            assert_eq!(
                families(&log),
                [
                    EffectFamily::Completion,
                    EffectFamily::Tool,
                    EffectFamily::Completion,
                    EffectFamily::Tool
                ]
            );
            crate::ecs_goldens::golden_effects(
                "anthropic_request_shape_tool_choice_required",
                &log,
            );
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
            let mut ecs = EcsAgent::for_golden(
                client.completion_model(CLAUDE_SONNET_4_6),
                TOOLS_PREAMBLE,
                false,
            );
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert(Temperature(Some(0.0)));
            ecs.tool(Adder);
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert(ToolChoiceSpec(Some(ToolChoice::Specific {
                    function_names: vec!["add".into()],
                })));
            let history = vec![];
            let run = spawn_run(
                ecs.app.world_mut(),
                ecs.agent,
                &history,
                ADD_PROMPT,
                false,
                Some(2),
            );
            let error = ecs
                .wait_for_outcome(run)
                .await
                .expect_err("forced tool choice exhausts budget");
            assert!(matches!(error, Failure::MaxTurns { limit: 2 }), "{error:?}");
            let log = ecs.effect_log();
            assert_eq!(
                families(&log),
                [
                    EffectFamily::Completion,
                    EffectFamily::Tool,
                    EffectFamily::Completion,
                    EffectFamily::Tool
                ]
            );
            crate::ecs_goldens::golden_effects(
                "anthropic_request_shape_tool_choice_specific",
                &log,
            );
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
            let mut ecs = EcsAgent::for_golden(client.completion_model(CLAUDE_SONNET_4_6), BASIC_PREAMBLE, false);
            ecs.app.world_mut().entity_mut(ecs.agent).insert(Temperature(Some(0.0)));
            ecs.tool(Adder);
            ecs.app.world_mut().entity_mut(ecs.agent).insert(ToolChoiceSpec(Some(ToolChoice::None)));
            let history = vec![];
            let run = spawn_run(ecs.app.world_mut(), ecs.agent, &history, NO_TOOL_PROMPT, false, Some(3));
            let output = ecs.wait_for_success(run).await;
            let log = ecs.effect_log();
            assert_eq!(families(&log), [EffectFamily::Completion]);
            let request = match &log.records[0].kind {
                rig::effect::EffectKind::Completion { request, .. } => request,
                other => panic!("a completion, not {other:?}"),
            };
            assert_eq!(request.tool_choice, Some(ToolChoice::None));
            assert_eq!(request.tools.len(), 1, "add is still advertised");
            assert_eq!(
                output, "",
                "Sonnet 4.6 answers `tool_choice: none` with empty content; if this changes, the cell changes"
            );
            crate::ecs_goldens::golden_effects("anthropic_request_shape_tool_choice_none", &log);
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
            let mut ecs = EcsAgent::for_golden(
                client.completion_model(CLAUDE_SONNET_4_6),
                BASIC_PREAMBLE,
                false,
            );
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert(Temperature(Some(0.0)));
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert(MaxTokens(Some(32)));
            let history = vec![];
            let run = spawn_run(
                ecs.app.world_mut(),
                ecs.agent,
                &history,
                BASIC_PROMPT,
                false,
                None,
            );
            let output = ecs.wait_for_success(run).await;
            assert!(!output.is_empty());
            let log = ecs.effect_log();
            assert_eq!(families(&log), [EffectFamily::Completion]);
            let request = match &log.records[0].kind {
                rig::effect::EffectKind::Completion { request, .. } => request,
                other => panic!("a completion, not {other:?}"),
            };
            assert_eq!(request.max_tokens, Some(32));
            crate::ecs_goldens::golden_effects("anthropic_request_shape_max_tokens", &log);
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
            let mut ecs = EcsAgent::for_golden(
                client.completion_model(CLAUDE_SONNET_4_6),
                BASIC_PREAMBLE,
                false,
            );
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert(AdditionalParams(Some(thinking_params())));
            let history = vec![];
            let run = spawn_run(
                ecs.app.world_mut(),
                ecs.agent,
                &history,
                THINKING_PROMPT,
                false,
                None,
            );
            let output = ecs.wait_for_success(run).await;
            assert!(output.contains("144"), "{}", output);
            let log = ecs.effect_log();
            assert_eq!(families(&log), [EffectFamily::Completion]);
            assert!(reasoning_blocks(&log) >= 1, "the completion reasons");
            crate::ecs_goldens::golden_effects("anthropic_request_shape_thinking_unary", &log);
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
            let mut ecs = EcsAgent::for_golden(
                client.completion_model(CLAUDE_SONNET_4_6),
                BASIC_PREAMBLE,
                true,
            );
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert(AdditionalParams(Some(thinking_params())));
            let history = vec![];
            let run = spawn_run(
                ecs.app.world_mut(),
                ecs.agent,
                &history,
                THINKING_PROMPT,
                true,
                None,
            );
            let output = ecs.wait_for_success(run).await;
            assert!(output.contains("144"), "{output}");
            let log = ecs.effect_log();
            assert_eq!(families(&log), [EffectFamily::Completion]);
            assert!(log.records[0].events.is_some(), "events are kept");
            assert!(reasoning_blocks(&log) >= 1, "the completion reasons");
            crate::ecs_goldens::golden_effects("anthropic_request_shape_thinking_streamed", &log);
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
            let mut ecs = EcsAgent::for_golden(
                client.completion_model(CLAUDE_SONNET_4_6),
                BASIC_PREAMBLE,
                false,
            );
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert(Temperature(Some(0.0)));
            for (n, text) in CONTEXT_DOCS.iter().take(2).enumerate() {
                let document = ecs
                    .app
                    .world_mut()
                    .spawn((
                        DocumentId(format!("static_doc_{n}")),
                        DocumentText((*text).into()),
                    ))
                    .id();
                ecs.app
                    .world_mut()
                    .spawn((Context(document), Order(n as u64), ChildOf(ecs.agent)));
            }
            let history = vec![];
            let run = spawn_run(
                ecs.app.world_mut(),
                ecs.agent,
                &history,
                CONTEXT_PROMPT,
                false,
                None,
            );
            let output = ecs.wait_for_success(run).await;
            assert!(!output.is_empty());
            let log = ecs.effect_log();
            assert_eq!(families(&log), [EffectFamily::Completion]);
            let request = match &log.records[0].kind {
                rig::effect::EffectKind::Completion { request, .. } => request,
                other => panic!("a completion, not {other:?}"),
            };
            assert_eq!(request.documents.len(), 2, "{:?}", request.documents);
            crate::ecs_goldens::golden_effects("anthropic_request_shape_static_context", &log);
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
            let mut ecs = EcsAgent::for_golden(
                client.completion_model(CLAUDE_SONNET_4_6),
                BASIC_PREAMBLE,
                false,
            );
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert(Temperature(Some(0.0)));
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert(Preamble(Some(format!(
                    "{BASIC_PREAMBLE}\nAlways end your answer with the word DONE."
                ))));
            let history = vec![];
            let run = spawn_run(
                ecs.app.world_mut(),
                ecs.agent,
                &history,
                BASIC_PROMPT,
                false,
                None,
            );
            let output = ecs.wait_for_success(run).await;
            assert!(output.contains("DONE"), "{}", output);
            let log = ecs.effect_log();
            assert_eq!(families(&log), [EffectFamily::Completion]);
            crate::ecs_goldens::golden_effects("anthropic_request_shape_append_preamble", &log);
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
            let mut ecs = EcsAgent::for_golden(
                client.completion_model(CLAUDE_SONNET_4_6),
                BASIC_PREAMBLE,
                false,
            );
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert(Temperature(Some(0.0)));
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert(Preamble(None));
            let history = vec![];
            let run = spawn_run(
                ecs.app.world_mut(),
                ecs.agent,
                &history,
                BASIC_PROMPT,
                false,
                None,
            );
            let output = ecs.wait_for_success(run).await;
            assert!(!output.is_empty());
            let log = ecs.effect_log();
            assert_eq!(families(&log), [EffectFamily::Completion]);
            let request = match &log.records[0].kind {
                rig::effect::EffectKind::Completion { request, .. } => request,
                other => panic!("a completion, not {other:?}"),
            };
            assert_eq!(request.system_instructions(), None);
            crate::ecs_goldens::golden_effects("anthropic_request_shape_without_preamble", &log);
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
            let mut ecs = EcsAgent::for_golden(
                client.completion_model(CLAUDE_SONNET_4_6),
                BASIC_PREAMBLE,
                false,
            );
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert(Temperature(Some(0.0)));
            ecs.app.world_mut().entity_mut(ecs.agent).insert(Output {
                mode: OutputKind::Auto,
                schema: Some(serde_json::from_str(EVENT_SCHEMA).expect("schema")),
            });
            let history = vec![];
            let run = spawn_run(
                ecs.app.world_mut(),
                ecs.agent,
                &history,
                STRUCTURED_OUTPUT_PROMPT,
                false,
                None,
            );
            let output = ecs.wait_for_success(run).await;
            let object: serde_json::Value =
                serde_json::from_str(&output).expect("the answer is the schema's object");
            assert!(object["title"].is_string(), "{object}");
            let log = ecs.effect_log();
            assert_eq!(families(&log), [EffectFamily::Completion]);
            assert_eq!(last_text(&log), output);
            crate::ecs_goldens::golden_effects("anthropic_request_shape_output_schema_unary", &log);
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
            let mut ecs = EcsAgent::for_golden(
                client.completion_model(CLAUDE_SONNET_4_6),
                BASIC_PREAMBLE,
                true,
            );
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert(Temperature(Some(0.0)));
            ecs.app.world_mut().entity_mut(ecs.agent).insert(Output {
                mode: OutputKind::Auto,
                schema: Some(serde_json::from_str(EVENT_SCHEMA).expect("schema")),
            });
            let history = vec![];
            let run = spawn_run(
                ecs.app.world_mut(),
                ecs.agent,
                &history,
                STRUCTURED_OUTPUT_PROMPT,
                true,
                None,
            );
            let output = ecs.wait_for_success(run).await;
            let object: serde_json::Value =
                serde_json::from_str(&output).expect("the answer is the schema's object");
            assert!(object["title"].is_string(), "{object}");
            let log = ecs.effect_log();
            assert_eq!(families(&log), [EffectFamily::Completion]);
            assert!(log.records[0].events.is_some(), "events are kept");
            crate::ecs_goldens::golden_effects(
                "anthropic_request_shape_output_schema_streamed",
                &log,
            );
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
            let mut ecs = EcsAgent::for_golden(
                client.completion_model(CLAUDE_SONNET_4_6),
                BASIC_PREAMBLE,
                false,
            );
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert(Temperature(Some(0.0)));
            let history = prior_history()
                .iter()
                .map(|message| MessageParts::from_message(message).expect("prior message"))
                .collect::<Vec<_>>();
            let run = spawn_run(
                ecs.app.world_mut(),
                ecs.agent,
                &history,
                NAME_PROMPT,
                false,
                None,
            );
            let output = ecs.wait_for_success(run).await;
            assert!(output.contains("Ada"), "{}", output);
            let log = ecs.effect_log();
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
            crate::ecs_goldens::golden_effects("anthropic_request_shape_prior_history", &log);
        },
    )
    .await;
}
