//! Matrix M of the effect corpus: per-turn shaping (`CLAUDE_SONNET_4_6`,
//! temperature 0, the route `CLAUDE_HAIKU_4_5`). A request patch from
//! `on_completion_call` shapes one turn's request and not the program; a
//! model-selection hook picks the turn's model. Producers of the goldens
//! `crates/rig-verify/tests/corpus_shaping.rs` replays by both
//! interpreters; the enumeration lives there. Every cell is a new
//! recording under `tests/cassettes/anthropic/corpus_shaping/`.

use futures::StreamExt;
use rig::agent::{Agent, MultiTurnStreamItem};
use rig::effect::{EffectFamily, EffectKind, HandlerKey};
use rig::message::ToolChoice;
use rig::prelude::*;
use rig::providers::anthropic::completion::{CLAUDE_HAIKU_4_5, CLAUDE_SONNET_4_6};
use rig::run::OutputMode;

use super::super::support::with_anthropic_corpus_shaping_cassette;
use crate::goldens::{
    LATE_ROUTE, PIRATE_PREAMBLE, PatchActiveToolsNoneSecond, PatchExtraContext, PatchHistoryFirst,
    PatchMaxTokensSecond, PatchPreambleSecond, PatchThinkingSecond, PatchToolChoiceNoneSecond,
    PatchToolChoiceRequiredFirst, PreambleOverride, RouteOnFirstTurn, SHAPING_CONTEXT, SelectLate,
    event_schema, families,
};
use crate::support::{Adder, BASIC_PREAMBLE, TOOLS_PREAMBLE};

const ADD_PROMPT: &str = "Use the add tool to add 17 and 25, then reply with just the number.";
const CONTEXT_PROMPT: &str = "What is a glarb-glarb? Answer in one sentence.";
const NAME_PROMPT: &str = "What is my name? Reply with just the name.";
const SUM_EVENT_PROMPT: &str = "Use the add tool to add 17 and 25, then return a concise event object for a Rust meetup in Seattle whose summary states the sum.";

fn request_at(log: &rig::effect_log::EffectLog, at: usize) -> &rig::completion::CompletionRequest {
    match &log.records[at].kind {
        EffectKind::Completion { request, .. } => request,
        other => panic!("record {at} is a completion, not {other:?}"),
    }
}

const TOOL_TURN: [EffectFamily; 3] = [
    EffectFamily::Completion,
    EffectFamily::Tool,
    EffectFamily::Completion,
];

async fn answer(agent: &Agent, prompt: &str) -> String {
    agent
        .prompt(prompt)
        .max_turns(3)
        .await
        .expect("the agent answers")
        .output
}

/// `tool_choice: Required` on turn 1 only: the second request is back to
/// the program's choice.
#[tokio::test]
async fn tool_choice_required_first_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_shaping_cassette(
        "corpus_shaping/tool_choice_required_first",
        |client| async move {
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .name("golden")
                .preamble(TOOLS_PREAMBLE)
                .temperature(0.0)
                .tool(Adder)
                .add_hook(PatchToolChoiceRequiredFirst)
                .record_effects()
                .build();
            let output = answer(&agent, ADD_PROMPT).await;
            assert!(output.contains("42"), "{output}");
            let log = agent.take_effect_log().expect("recording");
            assert_eq!(families(&log), TOOL_TURN);
            assert_eq!(request_at(&log, 0).tool_choice, Some(ToolChoice::Required));
            assert_eq!(request_at(&log, 2).tool_choice, None);
            crate::goldens::golden_effects("anthropic_shaping_tool_choice_required_first", &log);
        },
    )
    .await;
}

/// `tool_choice: None` on turn 2 of a committed `Tool`-mode run: the turn
/// cannot call the output tool (the engine warns and proceeds); what the
/// run then does is the record.
#[tokio::test]
async fn tool_choice_none_on_committed_output_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_shaping_cassette(
        "corpus_shaping/tool_choice_none_on_committed_output",
        |client| async move {
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .name("golden")
                .preamble(TOOLS_PREAMBLE)
                .temperature(0.0)
                .tool(Adder)
                .output_schema_raw(event_schema())
                .output_mode(OutputMode::Tool)
                .add_hook(PatchToolChoiceNoneSecond)
                .record_effects()
                .build();
            let outcome = agent.prompt(SUM_EVENT_PROMPT).max_turns(3).await;
            let log = agent.take_effect_log().expect("recording");
            assert_eq!(&families(&log)[..3], TOOL_TURN);
            assert_eq!(request_at(&log, 2).tool_choice, Some(ToolChoice::None));
            eprintln!("OUTCOME {outcome:?} RECORDS {}", log.records.len());
            crate::goldens::golden_effects(
                "anthropic_shaping_tool_choice_none_on_committed_output",
                &log,
            );
        },
    )
    .await;
}

/// A context document patched into every turn's request.
#[tokio::test]
async fn extra_context_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_shaping_cassette("corpus_shaping/extra_context", |client| async move {
        let agent = client
            .agent(CLAUDE_SONNET_4_6)
            .name("golden")
            .preamble(BASIC_PREAMBLE)
            .temperature(0.0)
            .add_hook(PatchExtraContext)
            .record_effects()
            .build();
        let output = answer(&agent, CONTEXT_PROMPT).await;
        assert!(output.to_lowercase().contains("jiro"), "{output}");
        let log = agent.take_effect_log().expect("recording");
        assert_eq!(families(&log), [EffectFamily::Completion]);
        assert!(
            request_at(&log, 0)
                .documents
                .iter()
                .any(|doc| doc.text == SHAPING_CONTEXT),
            "the patched document is in the request"
        );
        crate::goldens::golden_effects("anthropic_shaping_extra_context", &log);
    })
    .await;
}

/// The same, streamed with events.
#[tokio::test]
async fn extra_context_streamed_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_shaping_cassette(
        "corpus_shaping/extra_context_streamed",
        |client| async move {
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .name("golden")
                .preamble(BASIC_PREAMBLE)
                .temperature(0.0)
                .add_hook(PatchExtraContext)
                .record_effects_with_events()
                .build();
            let mut stream = agent.stream_prompt(CONTEXT_PROMPT).stream().await;
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
            assert!(output.to_lowercase().contains("jiro"), "{output}");
            let log = agent.take_effect_log().expect("recording");
            assert!(log.records[0].events.is_some(), "events are kept");
            crate::goldens::golden_effects("anthropic_shaping_extra_context_streamed", &log);
        },
    )
    .await;
}

/// Three hooks' patches merged in registration order: a preamble, a
/// document, a first-turn tool choice.
#[tokio::test]
async fn merged_three_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_shaping_cassette("corpus_shaping/merged_three", |client| async move {
        let agent = client
            .agent(CLAUDE_SONNET_4_6)
            .name("golden")
            .preamble(TOOLS_PREAMBLE)
            .temperature(0.0)
            .tool(Adder)
            .add_hook(PreambleOverride)
            .add_hook(PatchExtraContext)
            .add_hook(PatchToolChoiceRequiredFirst)
            .record_effects()
            .build();
        let output = answer(&agent, ADD_PROMPT).await;
        assert!(output.contains("42"), "{output}");
        let log = agent.take_effect_log().expect("recording");
        assert_eq!(families(&log), TOOL_TURN);
        let first = request_at(&log, 0);
        assert!(
            first
                .system_instructions()
                .is_some_and(|system| system.starts_with(PIRATE_PREAMBLE)),
            "the patched preamble"
        );
        assert!(
            first
                .documents
                .iter()
                .any(|doc| doc.text == SHAPING_CONTEXT)
        );
        assert_eq!(first.tool_choice, Some(ToolChoice::Required));
        assert_eq!(request_at(&log, 2).tool_choice, None);
        crate::goldens::golden_effects("anthropic_shaping_merged_three", &log);
    })
    .await;
}

/// The route selected on the first turn only: `fast` asks, the default
/// answers.
#[tokio::test]
async fn route_on_first_turn_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_shaping_cassette(
        "corpus_shaping/route_on_first_turn",
        |client| async move {
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .name("golden")
                .model_route("fast", client.completion_model(CLAUDE_HAIKU_4_5))
                .preamble(TOOLS_PREAMBLE)
                .temperature(0.0)
                .tool(Adder)
                .add_hook(RouteOnFirstTurn)
                .record_effects()
                .build();
            let output = answer(&agent, ADD_PROMPT).await;
            assert!(output.contains("42"), "{output}");
            let log = agent.take_effect_log().expect("recording");
            assert_eq!(families(&log), TOOL_TURN);
            assert_eq!(log.records[0].key, HandlerKey::from("golden/model:fast"));
            assert_eq!(log.records[2].key, HandlerKey::from("golden/model:default"));
            crate::goldens::golden_effects("anthropic_shaping_route_on_first_turn", &log);
        },
    )
    .await;
}

/// A route registered after build (`register_model`) and selected on
/// every turn: served, recorded, in the signature and the handler table,
/// and not in the required row (the row is the builder's).
#[tokio::test]
async fn late_route_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_shaping_cassette("corpus_shaping/late_route", |client| async move {
        let agent = client
            .agent(CLAUDE_SONNET_4_6)
            .name("golden")
            .preamble(TOOLS_PREAMBLE)
            .temperature(0.0)
            .tool(Adder)
            .add_hook(SelectLate)
            .record_effects()
            .build();
        agent.register_model(LATE_ROUTE, client.completion_model(CLAUDE_HAIKU_4_5));
        let output = answer(&agent, ADD_PROMPT).await;
        assert!(output.contains("42"), "{output}");
        let log = agent.take_effect_log().expect("recording");
        assert_eq!(families(&log), TOOL_TURN);
        let late = HandlerKey::from("golden/model:late");
        assert_eq!(log.records[0].key, late);
        assert!(
            !log.header.required.contains_key(&late),
            "{:?}",
            log.header.required
        );
        assert!(log.header.signature.contains_key(&late));
        assert!(
            log.header
                .handlers
                .iter()
                .any(|handler| handler.key == late)
        );
        crate::goldens::golden_effects("anthropic_shaping_late_route", &log);
    })
    .await;
}

/// `max_tokens: 5` on turn 2: the answer is cut where the patch says.
#[tokio::test]
async fn max_tokens_second_turn_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_shaping_cassette(
        "corpus_shaping/max_tokens_second_turn",
        |client| async move {
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .name("golden")
                .preamble(TOOLS_PREAMBLE)
                .temperature(0.0)
                .tool(Adder)
                .add_hook(PatchMaxTokensSecond)
                .record_effects()
                .build();
            let _ = answer(&agent, ADD_PROMPT).await;
            let log = agent.take_effect_log().expect("recording");
            assert_eq!(families(&log), TOOL_TURN);
            assert_eq!(request_at(&log, 0).max_tokens, None);
            assert_eq!(request_at(&log, 2).max_tokens, Some(5));
            crate::goldens::golden_effects("anthropic_shaping_max_tokens_second_turn", &log);
        },
    )
    .await;
}

/// Extended thinking on turn 2 only (with the temperature it needs).
#[tokio::test]
async fn thinking_second_turn_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_shaping_cassette(
        "corpus_shaping/thinking_second_turn",
        |client| async move {
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .name("golden")
                .preamble(TOOLS_PREAMBLE)
                .temperature(0.0)
                .tool(Adder)
                .add_hook(PatchThinkingSecond)
                .record_effects()
                .build();
            let output = answer(&agent, ADD_PROMPT).await;
            assert!(output.contains("42"), "{output}");
            let log = agent.take_effect_log().expect("recording");
            assert_eq!(families(&log), TOOL_TURN);
            assert!(request_at(&log, 0).additional_params.is_none());
            assert!(request_at(&log, 2).additional_params.is_some());
            assert_eq!(request_at(&log, 2).temperature, Some(1.0));
            crate::goldens::golden_effects("anthropic_shaping_thinking_second_turn", &log);
        },
    )
    .await;
}

/// The pirate preamble on turn 2 only.
#[tokio::test]
async fn preamble_second_turn_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_shaping_cassette(
        "corpus_shaping/preamble_second_turn",
        |client| async move {
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .name("golden")
                .preamble(TOOLS_PREAMBLE)
                .temperature(0.0)
                .tool(Adder)
                .add_hook(PatchPreambleSecond)
                .record_effects()
                .build();
            let _ = answer(&agent, ADD_PROMPT).await;
            let log = agent.take_effect_log().expect("recording");
            assert_eq!(families(&log), TOOL_TURN);
            assert!(
                request_at(&log, 0)
                    .system_instructions()
                    .is_some_and(|system| system.starts_with(TOOLS_PREAMBLE))
            );
            assert!(
                request_at(&log, 2)
                    .system_instructions()
                    .is_some_and(|system| system.starts_with(PIRATE_PREAMBLE))
            );
            crate::goldens::golden_effects("anthropic_shaping_preamble_second_turn", &log);
        },
    )
    .await;
}

/// No tools advertised on turn 2 (`active_tools: []`): the answer turn
/// sees none.
#[tokio::test]
async fn active_tools_none_second_turn_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_shaping_cassette(
        "corpus_shaping/active_tools_none_second_turn",
        |client| async move {
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .name("golden")
                .preamble(TOOLS_PREAMBLE)
                .temperature(0.0)
                .tool(Adder)
                .add_hook(PatchActiveToolsNoneSecond)
                .record_effects()
                .build();
            let output = answer(&agent, ADD_PROMPT).await;
            assert!(output.contains("42"), "{output}");
            let log = agent.take_effect_log().expect("recording");
            assert_eq!(families(&log), TOOL_TURN);
            assert_eq!(request_at(&log, 0).tools.len(), 1);
            assert!(request_at(&log, 2).tools.is_empty());
            crate::goldens::golden_effects("anthropic_shaping_active_tools_none_second_turn", &log);
        },
    )
    .await;
}

/// A prior exchange patched in as turn 1's history.
#[tokio::test]
async fn history_first_turn_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_shaping_cassette(
        "corpus_shaping/history_first_turn",
        |client| async move {
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .name("golden")
                .preamble(BASIC_PREAMBLE)
                .temperature(0.0)
                .add_hook(PatchHistoryFirst)
                .record_effects()
                .build();
            let output = answer(&agent, NAME_PROMPT).await;
            assert!(output.contains("Ada"), "{output}");
            let log = agent.take_effect_log().expect("recording");
            assert_eq!(families(&log), [EffectFamily::Completion]);
            assert!(
                request_at(&log, 0).chat_history.len() >= 3,
                "the patched exchange precedes the prompt: {:?}",
                request_at(&log, 0).chat_history
            );
            crate::goldens::golden_effects("anthropic_shaping_history_first_turn", &log);
        },
    )
    .await;
}
