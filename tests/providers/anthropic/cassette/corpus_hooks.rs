//! Matrix B of the effect corpus: the hook surface. Hooks are program: the
//! header names the stack, every decision is re-made on replay, and each
//! cell pins where a decision lands — in the record (a patched call, a
//! hook's own dispatch), in the transcript only (a replaced result, a
//! denied call), or in the request (an overridden preamble). Producers of
//! the goldens `crates/rig-verify/tests/corpus_hooks.rs` replays by both
//! interpreters; the enumeration lives there. The hooks themselves are in
//! `tests/common/goldens.rs`.
//!
//! Every cell is recorded once against the real Anthropic wire
//! (`CLAUDE_SONNET_4_6`, temperature 0) under
//! `tests/cassettes/anthropic/corpus_hooks/`.

use futures::StreamExt;
use rig::agent::MultiTurnStreamItem;
use rig::effect::{EffectFamily, EffectKind, Outcome};
use rig::message::{AssistantContent, Message, UserContent};
use rig::prelude::*;
use rig::providers::anthropic::completion::CLAUDE_SONNET_4_6;

use super::super::support::with_anthropic_corpus_hooks_cassette;
use crate::goldens::{
    DENY_REASON, DemandDone, DenyAdd, LOOKUP_ARGS, LookupBeforeRun, ObserveEverything,
    PIRATE_PREAMBLE, PatchAddArgs, PreambleOverride, REPLACED_ANSWER, REPLACED_RESULT,
    ReplaceAddResult, ReplaceAnswer, families,
};
use crate::support::{Adder, BASIC_PREAMBLE, BASIC_PROMPT, TOOLS_PREAMBLE};

const ADD_PROMPT: &str = "Use the add tool to add 17 and 25, then reply with just the number.";

fn request_at(
    log: &rig::effect_log::EffectLog,
    index: usize,
) -> &rig::completion::CompletionRequest {
    match &log.records[index].kind {
        EffectKind::Completion { request, .. } => request,
        other => panic!("record {index} is a completion, not {other:?}"),
    }
}

/// The text of every tool result in a request's history, in order.
fn tool_result_texts(request: &rig::completion::CompletionRequest) -> Vec<String> {
    request
        .chat_history
        .iter()
        .filter_map(|message| match message {
            Message::User { content } => Some(content.iter()),
            _ => None,
        })
        .flatten()
        .filter_map(|content| match content {
            UserContent::ToolResult(result) => Some(
                result
                    .content
                    .iter()
                    .map(|part| match part {
                        rig::message::ToolResultContent::Text(text) => text.text.clone(),
                        rig::message::ToolResultContent::Json { value } => value.to_string(),
                        other => format!("{other:?}"),
                    })
                    .collect::<String>(),
            ),
            _ => None,
        })
        .collect()
}

fn tool_record_args(log: &rig::effect_log::EffectLog) -> Vec<String> {
    log.records
        .iter()
        .filter_map(|record| match &record.kind {
            EffectKind::ToolCall { args, .. } => Some(args.clone()),
            _ => None,
        })
        .collect()
}

fn tool_record_outputs(log: &rig::effect_log::EffectLog) -> Vec<String> {
    log.records
        .iter()
        .filter_map(|record| match &record.outcome {
            Ok(Outcome::ToolResult { result, .. }) => Some(result.output().render()),
            _ => None,
        })
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

/// An observe-only hook opted into every family, over memory and a tool:
/// it sees the memory dispatches and changes nothing. The header names it.
#[tokio::test]
async fn observe_everything_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_hooks_cassette("corpus_hooks/observe_everything", |client| async move {
        let agent = client
            .agent(CLAUDE_SONNET_4_6)
            .name("golden")
            .preamble(TOOLS_PREAMBLE)
            .temperature(0.0)
            .tool(Adder)
            .memory(rig::memory::InMemoryConversationMemory::new())
            .conversation("golden-conversation")
            .add_hook(ObserveEverything)
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
                EffectFamily::Memory,
                EffectFamily::Completion,
                EffectFamily::Tool,
                EffectFamily::Completion,
                EffectFamily::Memory,
            ]
        );
        assert_eq!(log.header.hooks, ["ObserveEverything"]);
        crate::goldens::golden_effects("anthropic_hooks_observe_everything", &log);
    })
    .await;
}

/// `on_dispatch` → `Patch`: the tool record holds the patched arguments,
/// the model's history keeps the call it made.
#[tokio::test]
async fn patch_tool_args_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_hooks_cassette("corpus_hooks/patch_tool_args", |client| async move {
        let agent = client
            .agent(CLAUDE_SONNET_4_6)
            .name("golden")
            .preamble(TOOLS_PREAMBLE)
            .temperature(0.0)
            .tool(Adder)
            .add_hook(PatchAddArgs)
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
        assert_eq!(tool_record_args(&log), [r#"{"x":40,"y":2}"#]);
        let history = request_at(&log, 2);
        let called = history
            .chat_history
            .iter()
            .find_map(|message| match message {
                Message::Assistant { content, .. } => content.iter().find_map(|c| match c {
                    AssistantContent::ToolCall(call) => Some(call.function.arguments.clone()),
                    _ => None,
                }),
                _ => None,
            })
            .expect("the model's call is in history");
        assert_eq!(called, serde_json::json!({"x": 17, "y": 25}));
        crate::goldens::golden_effects("anthropic_hooks_patch_tool_args", &log);
    })
    .await;
}

/// The same, streamed with events kept.
#[tokio::test]
async fn patch_tool_args_streamed_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_hooks_cassette(
        "corpus_hooks/patch_tool_args_streamed",
        |client| async move {
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .name("golden")
                .preamble(TOOLS_PREAMBLE)
                .temperature(0.0)
                .tool(Adder)
                .add_hook(PatchAddArgs)
                .record_effects_with_events()
                .build();
            let mut stream = agent.stream_prompt(ADD_PROMPT).max_turns(3).stream().await;
            let output = final_output(&mut stream).await;
            drop(stream);
            assert!(output.contains("42"), "{output}");
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
            assert_eq!(tool_record_args(&log), [r#"{"x":40,"y":2}"#]);
            crate::goldens::golden_effects("anthropic_hooks_patch_tool_args_streamed", &log);
        },
    )
    .await;
}

/// `on_dispatch` → `Deny`: no tool record; the model sees the reason as
/// the tool's result and answers without it.
#[tokio::test]
async fn deny_tool_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_hooks_cassette("corpus_hooks/deny_tool", |client| async move {
        let agent = client
            .agent(CLAUDE_SONNET_4_6)
            .name("golden")
            .preamble(TOOLS_PREAMBLE)
            .temperature(0.0)
            .tool(Adder)
            .add_hook(DenyAdd)
            .record_effects()
            .build();
        let response = agent
            .prompt(ADD_PROMPT)
            .max_turns(3)
            .await
            .expect("the agent answers");
        assert!(!response.output.is_empty());
        let log = agent.take_effect_log().expect("recording");
        assert_eq!(
            families(&log),
            [EffectFamily::Completion, EffectFamily::Completion]
        );
        assert_eq!(tool_result_texts(request_at(&log, 1)), [DENY_REASON]);
        crate::goldens::golden_effects("anthropic_hooks_deny_tool", &log);
    })
    .await;
}

/// The same, streamed with events kept.
#[tokio::test]
async fn deny_tool_streamed_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_hooks_cassette("corpus_hooks/deny_tool_streamed", |client| async move {
        let agent = client
            .agent(CLAUDE_SONNET_4_6)
            .name("golden")
            .preamble(TOOLS_PREAMBLE)
            .temperature(0.0)
            .tool(Adder)
            .add_hook(DenyAdd)
            .record_effects_with_events()
            .build();
        let mut stream = agent.stream_prompt(ADD_PROMPT).max_turns(3).stream().await;
        let output = final_output(&mut stream).await;
        drop(stream);
        assert!(!output.is_empty());
        let log = agent.take_effect_log().expect("recording");
        assert_eq!(
            families(&log),
            [EffectFamily::Completion, EffectFamily::Completion]
        );
        assert!(log.records[0].events.is_some(), "events are kept");
        assert_eq!(tool_result_texts(request_at(&log, 1)), [DENY_REASON]);
        crate::goldens::golden_effects("anthropic_hooks_deny_tool_streamed", &log);
    })
    .await;
}

/// `on_outcome` → `Replace` on a tool: the record holds the tool's answer,
/// the transcript the replacement, and the model answers from the latter.
#[tokio::test]
async fn replace_tool_result_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_hooks_cassette("corpus_hooks/replace_tool_result", |client| async move {
        let agent = client
            .agent(CLAUDE_SONNET_4_6)
            .name("golden")
            .preamble(TOOLS_PREAMBLE)
            .temperature(0.0)
            .tool(Adder)
            .add_hook(ReplaceAddResult)
            .record_effects()
            .build();
        let response = agent
            .prompt(ADD_PROMPT)
            .max_turns(3)
            .await
            .expect("the agent answers");
        assert!(
            response.output.contains(REPLACED_RESULT),
            "{}",
            response.output
        );
        let log = agent.take_effect_log().expect("recording");
        assert_eq!(
            families(&log),
            [
                EffectFamily::Completion,
                EffectFamily::Tool,
                EffectFamily::Completion
            ]
        );
        assert_eq!(tool_record_outputs(&log), ["42"]);
        assert_eq!(tool_result_texts(request_at(&log, 2)), [REPLACED_RESULT]);
        crate::goldens::golden_effects("anthropic_hooks_replace_tool_result", &log);
    })
    .await;
}

/// `on_outcome` → `Replace` on a completion: the run's output is the
/// replacement, the record the model's text.
#[tokio::test]
async fn replace_answer_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_hooks_cassette("corpus_hooks/replace_answer", |client| async move {
        let agent = client
            .agent(CLAUDE_SONNET_4_6)
            .name("golden")
            .preamble(BASIC_PREAMBLE)
            .temperature(0.0)
            .add_hook(ReplaceAnswer)
            .record_effects()
            .build();
        let response = agent.prompt(BASIC_PROMPT).await.expect("the agent answers");
        assert_eq!(response.output, REPLACED_ANSWER);
        let log = agent.take_effect_log().expect("recording");
        assert_eq!(families(&log), [EffectFamily::Completion]);
        let recorded = match &log.records[0].outcome {
            Ok(Outcome::Completion(response)) => response,
            other => panic!("a completion, not {other:?}"),
        };
        assert!(
            !recorded
                .choice
                .iter()
                .any(|c| matches!(c, AssistantContent::Text(t) if t.text == REPLACED_ANSWER)),
            "the record holds the model's answer, not the replacement"
        );
        crate::goldens::golden_effects("anthropic_hooks_replace_answer", &log);
    })
    .await;
}

/// `on_completion_call` → a request patch: the request's system prompt is
/// the hook's, the spec's preamble is the builder's.
#[tokio::test]
async fn preamble_override_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_hooks_cassette("corpus_hooks/preamble_override", |client| async move {
        let agent = client
            .agent(CLAUDE_SONNET_4_6)
            .name("golden")
            .preamble(BASIC_PREAMBLE)
            .temperature(0.0)
            .add_hook(PreambleOverride)
            .record_effects()
            .build();
        let response = agent.prompt(BASIC_PROMPT).await.expect("the agent answers");
        assert!(!response.output.is_empty());
        let log = agent.take_effect_log().expect("recording");
        assert_eq!(families(&log), [EffectFamily::Completion]);
        assert_eq!(
            request_at(&log, 0).system_instructions(),
            Some(PIRATE_PREAMBLE)
        );
        assert_eq!(agent.run_spec().preamble.as_deref(), Some(BASIC_PREAMBLE));
        crate::goldens::golden_effects("anthropic_hooks_preamble_override", &log);
    })
    .await;
}

/// `on_model_turn_finished` → `Retry` with feedback: the first answer lacks
/// `DONE`, the hook asks again, the second has it. Two completions.
#[tokio::test]
async fn demand_done_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_hooks_cassette("corpus_hooks/demand_done", |client| async move {
        let agent = client
            .agent(CLAUDE_SONNET_4_6)
            .name("golden")
            .preamble(BASIC_PREAMBLE)
            .temperature(0.0)
            .add_hook(DemandDone)
            .record_effects()
            .build();
        let response = agent
            .prompt(BASIC_PROMPT)
            .max_turns(3)
            .await
            .expect("the agent answers");
        assert!(response.output.contains("DONE"), "{}", response.output);
        let log = agent.take_effect_log().expect("recording");
        assert_eq!(
            families(&log),
            [EffectFamily::Completion, EffectFamily::Completion]
        );
        crate::goldens::golden_effects("anthropic_hooks_demand_done", &log);
    })
    .await;
}

/// A hook that dispatches through the run's bus in `on_run_start`: the
/// hook's own tool call is the first record, under the tool's key.
#[tokio::test]
async fn lookup_before_run_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_hooks_cassette("corpus_hooks/lookup_before_run", |client| async move {
        let agent = client
            .agent(CLAUDE_SONNET_4_6)
            .name("golden")
            .preamble(TOOLS_PREAMBLE)
            .temperature(0.0)
            .tool(Adder)
            .add_hook(LookupBeforeRun)
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
                EffectFamily::Tool,
                EffectFamily::Completion,
                EffectFamily::Tool,
                EffectFamily::Completion
            ]
        );
        assert_eq!(tool_record_args(&log)[0], LOOKUP_ARGS);
        assert_eq!(log.records[0].key.as_str(), crate::goldens::LOOKUP_KEY);
        crate::goldens::golden_effects("anthropic_hooks_lookup_before_run", &log);
    })
    .await;
}

/// Two hooks in a stack: the header names both in registration order, and
/// both decisions land (the patched call in the record, the replaced
/// result in the transcript).
#[tokio::test]
async fn two_hooks_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_hooks_cassette("corpus_hooks/two_hooks", |client| async move {
        let agent = client
            .agent(CLAUDE_SONNET_4_6)
            .name("golden")
            .preamble(TOOLS_PREAMBLE)
            .temperature(0.0)
            .tool(Adder)
            .add_hook(PatchAddArgs)
            .add_hook(ReplaceAddResult)
            .record_effects()
            .build();
        let response = agent
            .prompt(ADD_PROMPT)
            .max_turns(3)
            .await
            .expect("the agent answers");
        assert!(
            response.output.contains(REPLACED_RESULT),
            "{}",
            response.output
        );
        let log = agent.take_effect_log().expect("recording");
        assert_eq!(log.header.hooks, ["PatchAddArgs", "ReplaceAddResult"]);
        assert_eq!(tool_record_args(&log), [r#"{"x":40,"y":2}"#]);
        assert_eq!(tool_record_outputs(&log), ["42"]);
        assert_eq!(tool_result_texts(request_at(&log, 2)), [REPLACED_RESULT]);
        crate::goldens::golden_effects("anthropic_hooks_two_hooks", &log);
    })
    .await;
}
