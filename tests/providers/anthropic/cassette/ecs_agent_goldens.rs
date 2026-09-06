//! Native provider-executed producers of the original completion/memory goldens.

use std::sync::Arc;

use rig::{
    effect::EffectFamily, memory::InMemoryConversationMemory, prelude::*,
    providers::anthropic::completion::CLAUDE_SONNET_4_6, serve::adapters::MemoryAdapter,
};
use rig_ecs::{
    agent::{Conversation, Remembers},
    bus::Handlers,
};

use super::super::support::with_anthropic_cassette;
use crate::{
    ecs_agent::{EcsAgent, RuntimeHandler},
    support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response},
};

#[tokio::test]
async fn completion_smoke_effect_log_is_the_golden_fixture() {
    with_anthropic_cassette("agent/completion_smoke", |client| async move {
        let mut ecs = EcsAgent::for_golden(
            client.completion_model(CLAUDE_SONNET_4_6),
            BASIC_PREAMBLE,
            false,
        );
        assert_nonempty_response(&ecs.prompt(BASIC_PROMPT, false).await);
        crate::ecs_goldens::golden_effects("anthropic_completion_smoke", &ecs.effect_log());
    })
    .await;
}

#[tokio::test]
async fn memory_conversation_effect_log_is_the_golden_fixture() {
    with_anthropic_cassette("agent/completion_smoke", |client| async move {
        let mut memory = None;
        let mut ecs = EcsAgent::for_golden_with_setup(
            client.completion_model(CLAUDE_SONNET_4_6),
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
                                runtime: tokio::runtime::Handle::current(),
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
        crate::ecs_goldens::golden_effects("anthropic_memory_conversation", &log);
    })
    .await;
}
