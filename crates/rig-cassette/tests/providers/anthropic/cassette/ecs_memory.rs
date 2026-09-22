//! Independent native memory producers using original provider fixtures/assertions.
use super::super::support::with_anthropic_corpus_memory_cassette;
use super::corpus_memory::{
    NAME_PROMPT, PROMPT, SECOND_PROMPT, bypass_history, loaded_lengths, memory_ops,
};
use crate::ecs_agent::EcsAgent;
use crate::goldens::{FailingMemory, families};
use crate::support::{
    AlphaSignal, BASIC_PREAMBLE, BetaSignal, TWO_TOOL_STREAM_PREAMBLE, TWO_TOOL_STREAM_PROMPT,
};
use rig::driver::Bound;
use rig::effect::{EffectFamily, EffectKind, HandlerKey, MemoryOp};
use rig::providers::anthropic::completion::CLAUDE_SONNET_4_6;
use rig::providers::anthropic::wire::Anthropic;
use rig_ecs::{
    agent::MessageParts,
    bus::{EffectOutcome, Policy},
    systems::RunCommands,
};
#[path = "ecs_memory/runtime.rs"]
mod runtime;
use runtime::*;

#[tokio::test]
async fn clear_at_start_effect_log() {
    crate::goldens::capture_world_programs(async {
        with_anthropic_corpus_memory_cassette(
            "corpus_memory/clear_at_start",
            |client| async move {
                let log = remembers(client, Clears::AtStart, &[PROMPT], false).await;
                crate::goldens::world_golden_effects(
                    "anthropic_memory_clear_at_start_effect_log",
                    &log,
                );
                // `on_run_start` fires after the load: the clear lands between the
                // load and the append.
                assert_eq!(memory_ops(&log), ["load", "clear", "append"]);
                assert_eq!(loaded_lengths(&log), [0]);
            },
        )
        .await;
    })
    .await
}

#[tokio::test]
async fn clear_at_settled_effect_log() {
    crate::goldens::capture_world_programs(async {
        with_anthropic_corpus_memory_cassette(
            "corpus_memory/clear_at_settled",
            |client| async move {
                let log = remembers(client, Clears::AtSettled, &[PROMPT], false).await;
                crate::goldens::world_golden_effects(
                    "anthropic_memory_clear_at_settled_effect_log",
                    &log,
                );
                assert_eq!(memory_ops(&log), ["load", "append", "clear"]);
            },
        )
        .await;
    })
    .await
}

/// Two runs over one conversation, one log: the second load holds the
/// first run's append.
#[tokio::test]
async fn two_runs_effect_log() {
    crate::goldens::capture_world_programs(async {
        with_anthropic_corpus_memory_cassette("corpus_memory/two_runs", |client| async move {
            let log = remembers(client, Clears::Never, &[PROMPT, SECOND_PROMPT], false).await;
            crate::goldens::world_golden_effects("anthropic_memory_two_runs_effect_log", &log);
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
        })
        .await;
    })
    .await
}

#[tokio::test]
async fn two_runs_streamed_effect_log() {
    crate::goldens::capture_world_programs(async {
        with_anthropic_corpus_memory_cassette(
            "corpus_memory/two_runs_streamed",
            |client| async move {
                let log = remembers(client, Clears::Never, &[PROMPT, SECOND_PROMPT], true).await;
                crate::goldens::world_golden_effects(
                    "anthropic_memory_two_runs_streamed_effect_log",
                    &log,
                );
                assert_eq!(memory_ops(&log), ["load", "append", "load", "append"]);
                assert_eq!(loaded_lengths(&log), [0, 2]);
                assert!(log.records[1].events.is_some(), "events are kept");
            },
        )
        .await;
    })
    .await
}

/// `Clear` after `Append`, twice: the second run loads nothing.
#[tokio::test]
async fn clear_at_settled_two_runs_effect_log() {
    crate::goldens::capture_world_programs(async {
        with_anthropic_corpus_memory_cassette(
            "corpus_memory/clear_at_settled_two_runs",
            |client| async move {
                let log =
                    remembers(client, Clears::AtSettled, &[PROMPT, SECOND_PROMPT], false).await;
                crate::goldens::world_golden_effects(
                    "anthropic_memory_clear_at_settled_two_runs_effect_log",
                    &log,
                );
                assert_eq!(
                    memory_ops(&log),
                    ["load", "append", "clear", "load", "append", "clear"]
                );
                assert_eq!(loaded_lengths(&log), [0, 0]);
            },
        )
        .await;
    })
    .await
}

/// `Clear` at run start, twice: the hook fires after the load, so the
/// second run still reads the first run's append before clearing it.
#[tokio::test]
async fn clear_at_start_two_runs_effect_log() {
    crate::goldens::capture_world_programs(async {
        with_anthropic_corpus_memory_cassette(
            "corpus_memory/clear_at_start_two_runs",
            |client| async move {
                let log = remembers(client, Clears::AtStart, &[PROMPT, SECOND_PROMPT], false).await;
                crate::goldens::world_golden_effects(
                    "anthropic_memory_clear_at_start_two_runs_effect_log",
                    &log,
                );
                assert_eq!(
                    memory_ops(&log),
                    ["load", "clear", "append", "load", "clear", "append"]
                );
                assert_eq!(loaded_lengths(&log), [0, 2]);
            },
        )
        .await;
    })
    .await
}

/// Explicit runner history bypasses memory: no `Load`, no `Append`.
#[tokio::test]
async fn history_bypass_effect_log() {
    crate::goldens::capture_world_programs(async {
        with_anthropic_corpus_memory_cassette(
            "corpus_memory/history_bypass",
            |client| async move {
                let mut ecs = agent(
                    &client,
                    rig_core::memory::InMemoryConversationMemory::new(),
                    BASIC_PREAMBLE,
                    false,
                );
                let history: Vec<_> = bypass_history()
                    .iter()
                    .map(|m| MessageParts::from_message(m).expect("history message"))
                    .collect();
                let run =
                    ecs.app
                        .world_mut()
                        .spawn_run(ecs.agent, &history, NAME_PROMPT, false, None);
                let output = ecs.wait_for_success(run).await;
                assert!(output.contains("Ada"), "{}", output);
                let log = ecs.effect_log();
                crate::goldens::world_golden_effects(
                    "anthropic_memory_history_bypass_effect_log",
                    &log,
                );
                assert_eq!(families(&log), [EffectFamily::Completion]);
                let program = log
                    .header
                    .programs
                    .values()
                    .next()
                    .expect("the run stamped its program identity");
                assert!(
                    program
                        .required
                        .contains_key(&HandlerKey::from("golden/memory")),
                    "memory is in the row though bypassed: {:?}",
                    program.required
                );
            },
        )
        .await;
    })
    .await
}

/// Memory over a host's bus: the builder registers the store on the
/// host's registrar under the agent's key, since only the builder can
/// name the run's memory.
#[tokio::test]
async fn host_bus_memory_effect_log() {
    crate::goldens::capture_world_programs(async {
        with_anthropic_corpus_memory_cassette(
            "corpus_memory/host_bus_memory",
            |client| async move {
                let mut ecs = EcsAgent::for_golden(
                    client.completion(CLAUDE_SONNET_4_6),
                    BASIC_PREAMBLE,
                    false,
                );
                let memory = register_memory(
                    ecs.app.world_mut(),
                    rig_core::memory::InMemoryConversationMemory::new(),
                );
                attach_memory(&mut ecs, memory);
                let output = ecs.prompt(PROMPT, false).await;
                assert!(!output.is_empty());
                let log = ecs.effect_log();
                crate::goldens::world_golden_effects(
                    "anthropic_memory_host_bus_memory_effect_log",
                    &log,
                );
                assert!(
                    ecs.app
                        .world_mut()
                        .query::<(&rig_ecs::bus::PendingEffect, Option<&EffectOutcome>)>()
                        .iter(ecs.app.world())
                        .all(|(_, outcome)| outcome.is_some()),
                    "host teardown has no unfinished effect"
                );
                drop(ecs);
                assert_eq!(memory_ops(&log), ["load", "append"]);
            },
        )
        .await;
    })
    .await
}

/// Serial serving, memory and two tool calls in one turn: the append
/// carries both results.
#[tokio::test]
async fn serial_two_tools_effect_log() {
    crate::goldens::capture_world_programs(async {
        with_anthropic_corpus_memory_cassette(
            "corpus_memory/serial_two_tools",
            |client| async move {
                let mut ecs = agent(
                    &client,
                    rig_core::memory::InMemoryConversationMemory::new(),
                    TWO_TOOL_STREAM_PREAMBLE,
                    true,
                );
                ecs.app
                    .world_mut()
                    .resource_mut::<Policy>()
                    .0
                    .serial_per_handler = true;
                ecs.tool(AlphaSignal);
                ecs.tool(BetaSignal);
                let outputs =
                    run_prompts(&mut ecs, &[TWO_TOOL_STREAM_PROMPT], true, Clears::Never).await;
                assert!(!outputs[0].is_empty());
                let log = ecs.effect_log();
                crate::goldens::world_golden_effects(
                    "anthropic_memory_serial_two_tools_effect_log",
                    &log,
                );
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
            },
        )
        .await;
    })
    .await
}

/// An `Append` that fails: the record holds the error and the run ends
/// in its answer regardless.
async fn append_fails(
    client: Bound<Anthropic>,
    streamed: bool,
) -> rig::cassette::effect_log::EffectLog {
    let mut ecs = agent(
        &client,
        FailingMemory::append_fails(),
        BASIC_PREAMBLE,
        streamed,
    );
    let outputs = run_prompts(&mut ecs, &[PROMPT], streamed, Clears::Never).await;
    assert!(!outputs[0].is_empty());
    let log = ecs.effect_log();
    assert_eq!(memory_ops(&log), ["load", "append"]);
    assert!(
        matches!(&log.records[2].outcome, Err(report) if report.kind == rig::error::ErrorKind::MemoryBackend),
        "{:?}",
        log.records[2].outcome
    );
    log
}

#[tokio::test]
async fn failing_append_effect_log() {
    crate::goldens::capture_world_programs(async {
        with_anthropic_corpus_memory_cassette(
            "corpus_memory/failing_append",
            |client| async move {
                let log = append_fails(client, false).await;
                crate::goldens::world_golden_effects(
                    "anthropic_memory_failing_append_effect_log",
                    &log,
                );
            },
        )
        .await;
    })
    .await
}

#[tokio::test]
async fn failing_append_streamed_effect_log() {
    crate::goldens::capture_world_programs(async {
        with_anthropic_corpus_memory_cassette(
            "corpus_memory/failing_append_streamed",
            |client| async move {
                let log = append_fails(client, true).await;
                crate::goldens::world_golden_effects(
                    "anthropic_memory_failing_append_streamed_effect_log",
                    &log,
                );
            },
        )
        .await;
    })
    .await
}
