//! Matrix J of the effect corpus: memory operations (`CLAUDE_SONNET_4_6`,
//! temperature 0, `InMemoryConversationMemory` under
//! `golden-conversation`). Producers of the goldens
//! `crates/rig-verify/tests/corpus_memory.rs` replays by both
//! interpreters; the enumeration lives there. Every cell is a new
//! recording under `tests/cassettes/anthropic/corpus_memory/`.

use futures::StreamExt;
use rig::agent::{AgentBuilder, MultiTurnStreamItem};
use rig::bus::{Bus, BusConfig};
use rig::effect::{EffectFamily, EffectKind, HandlerKey, MemoryOp};
use rig::message::Message;
use rig::prelude::*;
use rig::providers::anthropic::completion::CLAUDE_SONNET_4_6;

use super::super::support::with_anthropic_corpus_memory_cassette;
use crate::goldens::{CONVERSATION, ClearAtSettled, ClearAtStart, FailingMemory, families};
use crate::support::{
    AlphaSignal, BASIC_PREAMBLE, BetaSignal, TWO_TOOL_STREAM_PREAMBLE, TWO_TOOL_STREAM_PROMPT,
};

const PROMPT: &str = "Reply with the single word: ready.";
const SECOND_PROMPT: &str = "Now reply with the single word: again.";
const NAME_PROMPT: &str = "What is my name? Reply with just the name.";

fn bypass_history() -> Vec<Message> {
    vec![
        Message::user("My name is Ada."),
        Message::assistant("Hello, Ada."),
    ]
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

/// The memory ops of `log`, in order.
fn memory_ops(log: &rig::effect_log::EffectLog) -> Vec<&'static str> {
    log.iter()
        .filter_map(|record| match &record.kind {
            EffectKind::Memory { op } => Some(match op {
                MemoryOp::Load { .. } => "load",
                MemoryOp::Append { .. } => "append",
                MemoryOp::Clear { .. } => "clear",
            }),
            _ => None,
        })
        .collect()
}

/// The messages each `Load` answered with, in order.
fn loaded_lengths(log: &rig::effect_log::EffectLog) -> Vec<usize> {
    log.iter()
        .filter_map(|record| match (&record.kind, &record.outcome) {
            (
                EffectKind::Memory {
                    op: MemoryOp::Load { .. },
                },
                Ok(rig::effect::Outcome::Memory(rig::effect::MemoryOutcome::Loaded { messages })),
            ) => Some(messages.len()),
            _ => None,
        })
        .collect()
}

/// Which hook clears, if any.
#[derive(Clone, Copy)]
enum Clears {
    Never,
    AtStart,
    AtSettled,
}

async fn run_prompts(agent: &rig::agent::Agent, prompts: &[&str], streamed: bool) -> Vec<String> {
    let mut outputs = Vec::new();
    for prompt in prompts {
        let output = if streamed {
            let mut stream = agent.stream_prompt(*prompt).max_turns(8).stream().await;
            let output = final_output(&mut stream).await;
            drop(stream);
            output
        } else {
            agent
                .prompt(*prompt)
                .max_turns(3)
                .await
                .expect("the agent answers")
                .output
        };
        outputs.push(output);
    }
    outputs
}

/// A memory program on the agent's own bus.
async fn remembers(
    client: rig::providers::anthropic::Client,
    clears: Clears,
    prompts: &[&str],
    streamed: bool,
) -> rig::effect_log::EffectLog {
    let builder = client
        .agent(CLAUDE_SONNET_4_6)
        .name("golden")
        .preamble(BASIC_PREAMBLE)
        .temperature(0.0)
        .memory(rig::memory::InMemoryConversationMemory::new())
        .conversation(CONVERSATION);
    let builder = match clears {
        Clears::Never => builder,
        Clears::AtStart => builder.add_hook(ClearAtStart),
        Clears::AtSettled => builder.add_hook(ClearAtSettled),
    };
    let agent = if streamed {
        builder.record_effects_with_events().build()
    } else {
        builder.record_effects().build()
    };
    let outputs = run_prompts(&agent, prompts, streamed).await;
    for output in &outputs {
        assert!(!output.is_empty());
    }
    agent.take_effect_log().expect("recording")
}

#[tokio::test]
async fn clear_at_start_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_memory_cassette("corpus_memory/clear_at_start", |client| async move {
        let log = remembers(client, Clears::AtStart, &[PROMPT], false).await;
        // `on_run_start` fires after the load: the clear lands between the
        // load and the append.
        assert_eq!(memory_ops(&log), ["load", "clear", "append"]);
        assert_eq!(loaded_lengths(&log), [0]);
        crate::goldens::golden_effects("anthropic_memory_clear_at_start", &log);
    })
    .await;
}

#[tokio::test]
async fn clear_at_settled_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_memory_cassette("corpus_memory/clear_at_settled", |client| async move {
        let log = remembers(client, Clears::AtSettled, &[PROMPT], false).await;
        assert_eq!(memory_ops(&log), ["load", "append", "clear"]);
        crate::goldens::golden_effects("anthropic_memory_clear_at_settled", &log);
    })
    .await;
}

/// Two runs over one conversation, one log: the second load holds the
/// first run's append.
#[tokio::test]
async fn two_runs_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_memory_cassette("corpus_memory/two_runs", |client| async move {
        let log = remembers(client, Clears::Never, &[PROMPT, SECOND_PROMPT], false).await;
        assert_eq!(memory_ops(&log), ["load", "append", "load", "append"]);
        assert_eq!(loaded_lengths(&log), [0, 2]);
        assert_eq!(
            families(&log),
            [
                EffectFamily::Memory,
                EffectFamily::Completion,
                EffectFamily::Memory,
                EffectFamily::Memory,
                EffectFamily::Completion,
                EffectFamily::Memory,
            ]
        );
        crate::goldens::golden_effects("anthropic_memory_two_runs", &log);
    })
    .await;
}

#[tokio::test]
async fn two_runs_streamed_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_memory_cassette("corpus_memory/two_runs_streamed", |client| async move {
        let log = remembers(client, Clears::Never, &[PROMPT, SECOND_PROMPT], true).await;
        assert_eq!(memory_ops(&log), ["load", "append", "load", "append"]);
        assert_eq!(loaded_lengths(&log), [0, 2]);
        assert!(log.records[1].events.is_some(), "events are kept");
        crate::goldens::golden_effects("anthropic_memory_two_runs_streamed", &log);
    })
    .await;
}

/// `Clear` after `Append`, twice: the second run loads nothing.
#[tokio::test]
async fn clear_at_settled_two_runs_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_memory_cassette(
        "corpus_memory/clear_at_settled_two_runs",
        |client| async move {
            let log = remembers(client, Clears::AtSettled, &[PROMPT, SECOND_PROMPT], false).await;
            assert_eq!(
                memory_ops(&log),
                ["load", "append", "clear", "load", "append", "clear"]
            );
            assert_eq!(loaded_lengths(&log), [0, 0]);
            crate::goldens::golden_effects("anthropic_memory_clear_at_settled_two_runs", &log);
        },
    )
    .await;
}

/// `Clear` at run start, twice: the hook fires after the load, so the
/// second run still reads the first run's append before clearing it.
#[tokio::test]
async fn clear_at_start_two_runs_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_memory_cassette(
        "corpus_memory/clear_at_start_two_runs",
        |client| async move {
            let log = remembers(client, Clears::AtStart, &[PROMPT, SECOND_PROMPT], false).await;
            assert_eq!(
                memory_ops(&log),
                ["load", "clear", "append", "load", "clear", "append"]
            );
            assert_eq!(loaded_lengths(&log), [0, 2]);
            crate::goldens::golden_effects("anthropic_memory_clear_at_start_two_runs", &log);
        },
    )
    .await;
}

/// Explicit runner history bypasses memory: no `Load`, no `Append`.
#[tokio::test]
async fn history_bypass_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_memory_cassette("corpus_memory/history_bypass", |client| async move {
        let agent = client
            .agent(CLAUDE_SONNET_4_6)
            .name("golden")
            .preamble(BASIC_PREAMBLE)
            .temperature(0.0)
            .memory(rig::memory::InMemoryConversationMemory::new())
            .conversation(CONVERSATION)
            .record_effects()
            .build();
        let response = agent
            .prompt(NAME_PROMPT)
            .history(bypass_history())
            .await
            .expect("the agent answers");
        assert!(response.output.contains("Ada"), "{}", response.output);
        let log = agent.take_effect_log().expect("recording");
        assert_eq!(families(&log), [EffectFamily::Completion]);
        assert!(
            log.header
                .required
                .contains_key(&HandlerKey::from("golden/memory")),
            "memory is in the row though bypassed: {:?}",
            log.header.required
        );
        crate::goldens::golden_effects("anthropic_memory_history_bypass", &log);
    })
    .await;
}

/// Memory over a host's bus: the builder registers the store on the
/// host's registrar under the agent's key, since only the builder can
/// name the run's memory.
#[tokio::test]
async fn host_bus_memory_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_memory_cassette("corpus_memory/host_bus_memory", |client| async move {
        let (dispatcher, registrar, mut driver) = Bus::channel();
        let model_key = HandlerKey::from("golden/model:default");
        driver
            .register_erased(
                model_key.clone(),
                rig::serve::ErasedHandler::new(rig::serve::adapters::CompletionAdapter::new(
                    "default",
                    client.completion_model(CLAUDE_SONNET_4_6),
                )),
            )
            .expect("a fresh key");
        let recorder = rig::effect_log::EffectLogRecorder::new();
        driver.record_to(recorder.clone());
        let driver = tokio::spawn(driver);
        let agent =
            AgentBuilder::over_bus(dispatcher.clone(), registrar.clone(), "golden", model_key)
                .name("golden")
                .preamble(BASIC_PREAMBLE)
                .temperature(0.0)
                .memory(rig::memory::InMemoryConversationMemory::new())
                .conversation(CONVERSATION)
                .build();
        let output = agent
            .prompt(PROMPT)
            .await
            .expect("the agent answers")
            .output;
        assert!(!output.is_empty());
        let log = agent.stamp(recorder.take());
        drop((agent, dispatcher, registrar));
        driver.await.expect("the host's driver");
        assert_eq!(log.header.bus, None);
        assert_eq!(memory_ops(&log), ["load", "append"]);
        crate::goldens::golden_effects("anthropic_memory_host_bus", &log);
    })
    .await;
}

/// Serial serving, memory and two tool calls in one turn: the append
/// carries both results.
#[tokio::test]
async fn serial_two_tools_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_memory_cassette("corpus_memory/serial_two_tools", |client| async move {
        let agent = client
            .agent(CLAUDE_SONNET_4_6)
            .name("golden")
            .configure_bus(BusConfig {
                serial_per_handler: true,
                ..BusConfig::default()
            })
            .preamble(TWO_TOOL_STREAM_PREAMBLE)
            .temperature(0.0)
            .tool(AlphaSignal)
            .tool(BetaSignal)
            .memory(rig::memory::InMemoryConversationMemory::new())
            .conversation(CONVERSATION)
            .record_effects_with_events()
            .build();
        let outputs = run_prompts(&agent, &[TWO_TOOL_STREAM_PROMPT], true).await;
        assert!(!outputs[0].is_empty());
        let log = agent.take_effect_log().expect("recording");
        assert_eq!(
            families(&log),
            [
                EffectFamily::Memory,
                EffectFamily::Completion,
                EffectFamily::Tool,
                EffectFamily::Tool,
                EffectFamily::Completion,
                EffectFamily::Memory,
            ]
        );
        let appended = match &log.records[5].kind {
            EffectKind::Memory {
                op: MemoryOp::Append { messages, .. },
            } => messages.len(),
            other => panic!("an append, not {other:?}"),
        };
        assert_eq!(appended, 4, "prompt, call turn, results, answer");
        crate::goldens::golden_effects("anthropic_memory_serial_two_tools", &log);
    })
    .await;
}

/// An `Append` that fails: the record holds the error and the run ends
/// in its answer regardless.
async fn append_fails(
    client: rig::providers::anthropic::Client,
    streamed: bool,
) -> rig::effect_log::EffectLog {
    let builder = client
        .agent(CLAUDE_SONNET_4_6)
        .name("golden")
        .preamble(BASIC_PREAMBLE)
        .temperature(0.0)
        .memory(FailingMemory::append_fails())
        .conversation(CONVERSATION);
    let agent = if streamed {
        builder.record_effects_with_events().build()
    } else {
        builder.record_effects().build()
    };
    let outputs = run_prompts(&agent, &[PROMPT], streamed).await;
    assert!(!outputs[0].is_empty());
    let log = agent.take_effect_log().expect("recording");
    assert_eq!(memory_ops(&log), ["load", "append"]);
    assert!(
        matches!(&log.records[2].outcome, Err(report) if report.kind == rig::error::ErrorKind::MemoryBackend),
        "{:?}",
        log.records[2].outcome
    );
    log
}

#[tokio::test]
async fn failing_append_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_memory_cassette("corpus_memory/failing_append", |client| async move {
        let log = append_fails(client, false).await;
        crate::goldens::golden_effects("anthropic_memory_failing_append", &log);
    })
    .await;
}

#[tokio::test]
async fn failing_append_streamed_effect_log_is_the_golden_fixture() {
    with_anthropic_corpus_memory_cassette(
        "corpus_memory/failing_append_streamed",
        |client| async move {
            let log = append_fails(client, true).await;
            crate::goldens::golden_effects("anthropic_memory_failing_append_streamed", &log);
        },
    )
    .await;
}
