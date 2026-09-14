//! Native output-mode counterparts using original assertions and provider cassettes.
use super::super::support::with_anthropic_corpus_output_cassette;
use super::corpus_output::{SUM_EVENT_PROMPT, assert_event, request_at, tool_names};
use crate::ecs_agent::EcsAgent;
use crate::goldens::{event_schema, families};
use crate::support::{Adder, BASIC_PREAMBLE, STRUCTURED_OUTPUT_PROMPT, TOOLS_PREAMBLE};
use rig::effect::EffectFamily;
use rig::message::ToolChoice;
use rig::prelude::*;
use rig::providers::anthropic::completion::CLAUDE_SONNET_4_6;
use rig_ecs::agent::*;

#[tokio::test]
async fn tool_unary_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_output_cassette("corpus_output/tool_unary", |client| async move {
        let mut ecs = EcsAgent::for_golden(
            client.completion_model(CLAUDE_SONNET_4_6),
            BASIC_PREAMBLE,
            false,
        );
        ecs.app.world_mut().entity_mut(ecs.agent).insert(Output {
            mode: OutputKind::Tool,
            schema: Some(event_schema().into()),
        });
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert(Temperature(Some(0.0)));

        let output = ecs
            .prompt_with_max_turns(STRUCTURED_OUTPUT_PROMPT, false, None)
            .await;
        assert_event(&output);
        let log = ecs.effect_log();
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
        crate::ecs_goldens::golden_effects("anthropic_output_tool_unary", &log);
    })
    .await;
}

/// The same, streamed with events.

#[tokio::test]
async fn tool_streamed_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_output_cassette("corpus_output/tool_streamed", |client| async move {
        let mut ecs = EcsAgent::for_golden(
            client.completion_model(CLAUDE_SONNET_4_6),
            BASIC_PREAMBLE,
            true,
        );
        ecs.app.world_mut().entity_mut(ecs.agent).insert(Output {
            mode: OutputKind::Tool,
            schema: Some(event_schema().into()),
        });
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert(Temperature(Some(0.0)));

        let output = ecs
            .prompt_with_max_turns(STRUCTURED_OUTPUT_PROMPT, true, None)
            .await;
        assert_event(&output);
        let log = ecs.effect_log();
        assert_eq!(families(&log), [EffectFamily::Completion]);
        assert!(log.records[0].events.is_some(), "events are kept");
        assert_eq!(tool_names(request_at(&log, 0)), ["final_result"]);
        crate::ecs_goldens::golden_effects("anthropic_output_tool_streamed", &log);
    })
    .await;
}

/// `Prompted` mode: no tool, no native schema; the preamble carries the
/// schema and the answer is JSON text.

#[tokio::test]
async fn prompted_unary_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_output_cassette("corpus_output/prompted_unary", |client| async move {
        let mut ecs = EcsAgent::for_golden(
            client.completion_model(CLAUDE_SONNET_4_6),
            BASIC_PREAMBLE,
            false,
        );
        ecs.app.world_mut().entity_mut(ecs.agent).insert(Output {
            mode: OutputKind::Prompted,
            schema: Some(event_schema().into()),
        });
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert(Temperature(Some(0.0)));

        let output = ecs
            .prompt_with_max_turns(STRUCTURED_OUTPUT_PROMPT, false, None)
            .await;
        assert_event(&output);
        let log = ecs.effect_log();
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
        crate::ecs_goldens::golden_effects("anthropic_output_prompted_unary", &log);
    })
    .await;
}

/// The same, streamed with events.

#[tokio::test]
async fn prompted_streamed_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_output_cassette("corpus_output/prompted_streamed", |client| async move {
        let mut ecs = EcsAgent::for_golden(
            client.completion_model(CLAUDE_SONNET_4_6),
            BASIC_PREAMBLE,
            true,
        );
        ecs.app.world_mut().entity_mut(ecs.agent).insert(Output {
            mode: OutputKind::Prompted,
            schema: Some(event_schema().into()),
        });
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert(Temperature(Some(0.0)));

        let output = ecs
            .prompt_with_max_turns(STRUCTURED_OUTPUT_PROMPT, true, None)
            .await;
        assert_event(&output);
        let log = ecs.effect_log();
        assert_eq!(families(&log), [EffectFamily::Completion]);
        assert!(log.records[0].events.is_some(), "events are kept");
        crate::ecs_goldens::golden_effects("anthropic_output_prompted_streamed", &log);
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
            let mut ecs = EcsAgent::for_golden(
                client.completion_model(CLAUDE_SONNET_4_6),
                TOOLS_PREAMBLE,
                false,
            );
            ecs.app.world_mut().entity_mut(ecs.agent).insert(Output {
                mode: OutputKind::Tool,
                schema: Some(event_schema().into()),
            });
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert(Temperature(Some(0.0)));
            ecs.tool(Adder);

            let output = ecs
                .prompt_with_max_turns(SUM_EVENT_PROMPT, false, Some(3))
                .await;
            assert_event(&output);
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
            assert_eq!(tool_names(request_at(&log, 0)), ["add", "final_result"]);
            crate::ecs_goldens::golden_effects("anthropic_output_tool_with_real_tool", &log);
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
            let mut ecs = EcsAgent::for_golden(
                client.completion_model(CLAUDE_SONNET_4_6),
                TOOLS_PREAMBLE,
                false,
            );
            ecs.app.world_mut().entity_mut(ecs.agent).insert(Output {
                mode: OutputKind::Prompted,
                schema: Some(event_schema().into()),
            });
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert(Temperature(Some(0.0)));
            ecs.tool(Adder);

            let output = ecs
                .prompt_with_max_turns(SUM_EVENT_PROMPT, false, Some(3))
                .await;
            assert_event(&output);
            let log = ecs.effect_log();
            assert_eq!(
                families(&log),
                [
                    EffectFamily::Completion,
                    EffectFamily::Tool,
                    EffectFamily::Completion
                ]
            );
            assert_eq!(tool_names(request_at(&log, 0)), ["add"]);
            crate::ecs_goldens::golden_effects("anthropic_output_prompted_with_real_tool", &log);
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
            let mut ecs = EcsAgent::for_golden(
                client.completion_model(CLAUDE_SONNET_4_6),
                BASIC_PREAMBLE,
                false,
            );
            ecs.app.world_mut().entity_mut(ecs.agent).insert(Output {
                mode: OutputKind::Tool,
                schema: Some(event_schema().into()),
            });
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert(Temperature(Some(0.0)));
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert(ToolChoiceSpec(Some(ToolChoice::Specific {
                    function_names: vec!["final_result".to_owned()],
                })));

            let output = ecs
                .prompt_with_max_turns(STRUCTURED_OUTPUT_PROMPT, false, None)
                .await;
            assert_event(&output);
            let log = ecs.effect_log();
            assert_eq!(families(&log), [EffectFamily::Completion]);
            assert_eq!(tool_names(request_at(&log, 0)), ["final_result"]);
            crate::ecs_goldens::golden_effects(
                "anthropic_output_tool_choice_specific_output",
                &log,
            );
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
            let mut ecs = EcsAgent::for_golden(
                client.completion_model(CLAUDE_SONNET_4_6),
                BASIC_PREAMBLE,
                false,
            );
            ecs.app.world_mut().entity_mut(ecs.agent).insert(Output {
                mode: OutputKind::Tool,
                schema: Some(event_schema().into()),
            });
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert(Temperature(Some(0.0)));
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert(ToolChoiceSpec(Some(ToolChoice::Required)));

            let output = ecs
                .prompt_with_max_turns(STRUCTURED_OUTPUT_PROMPT, false, Some(2))
                .await;
            assert_event(&output);
            let log = ecs.effect_log();
            assert_eq!(families(&log), [EffectFamily::Completion]);
            crate::ecs_goldens::golden_effects("anthropic_output_tool_choice_required", &log);
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
            let mut ecs = EcsAgent::for_golden(
                client.completion_model(CLAUDE_SONNET_4_6),
                BASIC_PREAMBLE,
                false,
            );
            ecs.app.world_mut().entity_mut(ecs.agent).insert(Output {
                mode: OutputKind::Tool,
                schema: Some(event_schema().into()),
            });
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert(Temperature(Some(0.0)));
            ecs.app
                .world_mut()
                .entity_mut(ecs.agent)
                .insert(ToolChoiceSpec(Some(ToolChoice::None)));

            let output = ecs
                .prompt_with_max_turns(STRUCTURED_OUTPUT_PROMPT, false, None)
                .await;
            assert_event(&output);
            let log = ecs.effect_log();
            assert_eq!(families(&log), [EffectFamily::Completion]);
            let request = request_at(&log, 0);
            assert!(
                request.tools.is_empty(),
                "no output tool under tool_choice none"
            );
            assert!(request.output_schema.is_some(), "the native schema instead");
            crate::ecs_goldens::golden_effects("anthropic_output_tool_under_none_degrades", &log);
        },
    )
    .await;
}

/// `Tool` mode under extended thinking: the record holds a reasoning
/// block and the output tool's call.

#[tokio::test]
async fn tool_thinking_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_output_cassette("corpus_output/tool_thinking", |client| async move {
        let mut ecs = EcsAgent::for_golden(
            client.completion_model(CLAUDE_SONNET_4_6),
            BASIC_PREAMBLE,
            false,
        );
        ecs.app.world_mut().entity_mut(ecs.agent).insert(Output {
            mode: OutputKind::Tool,
            schema: Some(event_schema().into()),
        });
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert(AdditionalParams(Some(
                serde_json::json!({"thinking":{"type":"enabled","budget_tokens":1024}}),
            )));

        let output = ecs
            .prompt_with_max_turns(STRUCTURED_OUTPUT_PROMPT, false, None)
            .await;
        assert_event(&output);
        let log = ecs.effect_log();
        assert_eq!(families(&log), [EffectFamily::Completion]);
        crate::ecs_goldens::golden_effects("anthropic_output_tool_thinking", &log);
    })
    .await;
}
