//! Native smoke cells over the completion and memory recordings: the world
//! answers, its record is one completion (with memory around it where the
//! agent remembers), and the wire reported usage.

use std::sync::Arc;

use rig::{
    effect::{EffectFamily, Outcome},
    memory::InMemoryConversationMemory,
    providers::anthropic::completion::CLAUDE_SONNET_4_6,
    serve::adapters::MemoryAdapter,
};
use rig_cassette::effect_log::EffectLog;
use rig_ecs::{
    agent::{Conversation, Remembers},
    bus::Handlers,
};

use super::super::support::with_anthropic_cassette;
use crate::{
    ecs_agent::{EcsAgent, RuntimeHandler, io_runtime},
    support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response},
};

#[tokio::test]
async fn completion_smoke_effect_log() {
    crate::goldens::capture_world_programs(async {
        with_anthropic_cassette("agent/completion_smoke", |client| async move {
            let mut ecs =
                EcsAgent::for_golden(client.completion(CLAUDE_SONNET_4_6), BASIC_PREAMBLE, false);
            assert_nonempty_response(&ecs.prompt(BASIC_PROMPT, false).await);
            let log = ecs.effect_log();
            crate::goldens::world_golden_effects(
                "anthropic_agent_smoke_completion_smoke_effect_log",
                &log,
            );
            assert_eq!(
                log.records
                    .iter()
                    .map(|record| record.kind.family())
                    .collect::<Vec<_>>(),
                [EffectFamily::Completion],
                "one completion"
            );
            assert_reported_usage(&log);
        })
        .await;
    })
    .await
}

#[tokio::test]
async fn memory_conversation_effect_log() {
    crate::goldens::capture_world_programs(async {
        with_anthropic_cassette("agent/completion_smoke", |client| async move {
            let mut memory = None;
            let mut ecs = EcsAgent::for_golden_with_setup(
                client.completion(CLAUDE_SONNET_4_6),
                BASIC_PREAMBLE,
                false,
                |world| {
                    memory = Some(
                        Handlers::with(world, |handlers| {
                            handlers.register(
                                "golden/memory",
                                RuntimeHandler {
                                    inner: Arc::new(MemoryAdapter::new(
                                        InMemoryConversationMemory::new(),
                                    )),
                                    runtime: io_runtime(),
                                },
                            )
                        })
                        .expect("bus installed")
                        .expect("fresh memory key"),
                    );
                },
            );
            ecs.app.world_mut().entity_mut(ecs.agent).insert((
                Remembers(memory.expect("setup registered memory")),
                Conversation("golden-conversation".into()),
            ));
            assert_nonempty_response(&ecs.prompt(BASIC_PROMPT, false).await);
            let log = ecs.effect_log();
            crate::goldens::world_golden_effects(
                "anthropic_agent_smoke_memory_conversation_effect_log",
                &log,
            );
            assert_eq!(
                log.records
                    .iter()
                    .map(|record| record.kind.family())
                    .collect::<Vec<_>>(),
                [
                    EffectFamily::Memory,
                    EffectFamily::Completion,
                    EffectFamily::Memory
                ],
                "load, completion, append"
            );
            assert_reported_usage(&log);
        })
        .await;
    })
    .await
}

/// Every completion record's response carries the usage the wire reported.
fn assert_reported_usage(log: &EffectLog) {
    let usages = log
        .records
        .iter()
        .filter_map(|record| match &record.outcome {
            Ok(Outcome::Completion(response)) => Some(&response.usage),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(!usages.is_empty(), "a completion record");
    for usage in usages {
        assert!(
            usage.input_tokens.is_some() && usage.output_tokens.is_some(),
            "the wire reported usage: {usage:?}"
        );
    }
}
