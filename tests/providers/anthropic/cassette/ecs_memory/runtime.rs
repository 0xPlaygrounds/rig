//! Native memory setup and application clear policies. Every operation uses the bus.
use crate::{
    ecs_agent::{EcsAgent, RuntimeHandler},
    goldens::{CONVERSATION, MEMORY_KEY},
    support::BASIC_PREAMBLE,
};
use bevy_ecs::prelude::*;
use rig::prelude::*;
use rig::providers::anthropic::{Client, completion::CLAUDE_SONNET_4_6};
use rig_core::{
    effect::{EffectKind, MemoryOp, MemoryOutcome, Outcome},
    memory::ConversationMemory,
    serve::adapters::MemoryAdapter,
};
use rig_ecs::{
    agent::{Conversation, Cursor, PolicyVersion, Remembers, Run, Settled, Temperature},
    bus::{Bound, EffectOutcome, Handlers, PendingEffect, RigSchedule},
    systems::{RigSet, spawn_run},
};
use std::sync::Arc;

#[derive(Clone, Copy)]
pub(super) enum Clears {
    Never,
    AtStart,
    AtSettled,
}
#[derive(Component)]
struct ClearEffect;

pub(super) fn register_memory(
    world: &mut World,
    memory: impl ConversationMemory + 'static,
) -> Entity {
    Handlers::with(world, |h| {
        h.register(
            MEMORY_KEY,
            RuntimeHandler {
                inner: Arc::new(MemoryAdapter::new(memory)),
                runtime: tokio::runtime::Handle::current(),
            },
        )
    })
    .expect("bus installed")
    .expect("memory key")
}
pub(super) fn agent(
    client: &Client,
    memory: impl ConversationMemory + 'static,
    preamble: &str,
    streamed: bool,
) -> EcsAgent {
    let mut ecs = EcsAgent::for_golden_with_setup(
        client.completion_model(CLAUDE_SONNET_4_6),
        preamble,
        streamed,
        |world| {
            register_memory(world, memory);
        },
    );
    let memory = ecs
        .app
        .world_mut()
        .query::<(Entity, &Bound)>()
        .iter(ecs.app.world())
        .find(|(_, b)| b.key.as_str() == MEMORY_KEY)
        .expect("memory")
        .0;
    attach_memory(&mut ecs, memory);
    ecs
}
pub(super) fn attach_memory(ecs: &mut EcsAgent, memory: Entity) {
    ecs.app.world_mut().entity_mut(ecs.agent).insert((
        Remembers(memory),
        Conversation(CONVERSATION.into()),
        Temperature(Some(0.0)),
    ));
}
fn clear(commands: &mut Commands, run: Entity) {
    commands.spawn((
        ClearEffect,
        PendingEffect::new(
            MEMORY_KEY,
            EffectKind::Memory {
                op: MemoryOp::Clear {
                    conversation: CONVERSATION.into(),
                },
            },
        ),
        ChildOf(run),
    ));
}
fn clear_after_load(
    added: On<Add, EffectOutcome>,
    effects: Query<(&PendingEffect, &EffectOutcome, &ChildOf)>,
    mut commands: Commands,
) {
    let Ok((effect, outcome, parent)) = effects.get(added.event().entity) else {
        return;
    };
    if matches!(
        effect.kind,
        EffectKind::Memory {
            op: MemoryOp::Load { .. }
        }
    ) && matches!(outcome.0, Ok(Outcome::Memory(MemoryOutcome::Loaded { .. })))
    {
        clear(&mut commands, parent.parent());
    }
}
fn clear_after_append(
    added: On<Add, EffectOutcome>,
    effects: Query<(&PendingEffect, &ChildOf)>,
    mut commands: Commands,
) {
    let Ok((effect, parent)) = effects.get(added.event().entity) else {
        return;
    };
    if matches!(
        effect.kind,
        EffectKind::Memory {
            op: MemoryOp::Append { .. }
        }
    ) {
        clear(&mut commands, parent.parent());
    }
}
fn cleared(outcome: &EffectOutcome) {
    assert!(
        matches!(outcome.0, Ok(Outcome::Memory(MemoryOutcome::Cleared))),
        "the memory clears: {:?}",
        outcome.0
    );
}
// These scenarios execute one active run at a time. Keep each awaited startup
// clear ahead of the first model turn, including when previous runs remain.
type ActiveRuns<'w, 's> = Query<'w, 's, (Entity, &'static Cursor), (With<Run>, Without<Settled>)>;
fn startup_cleared(
    runs: ActiveRuns,
    effects: Query<(&ChildOf, Option<&EffectOutcome>), With<ClearEffect>>,
) -> bool {
    for (run, cursor) in &runs {
        if cursor.turn == 0 {
            let Some((_, Some(outcome))) =
                effects.iter().find(|(parent, _)| parent.parent() == run)
            else {
                return false;
            };
            cleared(outcome);
        }
    }
    true
}
async fn await_clear(ecs: &mut EcsAgent, run: Entity) {
    tokio::time::timeout(std::time::Duration::from_secs(30), async {
        loop {
            ecs.app.update();
            let world = ecs.app.world_mut();
            let mut query =
                world.query_filtered::<(&ChildOf, Option<&EffectOutcome>), With<ClearEffect>>();
            let matching: Vec<_> = query
                .iter(world)
                .filter(|(parent, _)| parent.parent() == run)
                .collect();
            assert!(matching.len() <= 1, "one clear per run");
            if let Some((_, Some(outcome))) = matching.first() {
                cleared(outcome);
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("clear acknowledgement before returning");
}
pub(super) async fn run_prompts(
    ecs: &mut EcsAgent,
    prompts: &[&str],
    streamed: bool,
    clears: Clears,
) -> Vec<String> {
    let mut outputs = vec![];
    for prompt in prompts {
        let run = spawn_run(
            ecs.app.world_mut(),
            ecs.agent,
            &[],
            prompt,
            streamed,
            Some(if streamed { 8 } else { 3 }),
        );
        let output = ecs.wait_for_success(run).await;
        if !matches!(clears, Clears::Never) {
            await_clear(ecs, run).await;
        }
        outputs.push(output);
    }
    outputs
}
pub(super) async fn remembers(
    client: Client,
    clears: Clears,
    prompts: &[&str],
    streamed: bool,
) -> rig_effect_log::EffectLog {
    let mut ecs = agent(
        &client,
        rig_core::memory::InMemoryConversationMemory::new(),
        BASIC_PREAMBLE,
        streamed,
    );
    match clears {
        Clears::Never => {}
        Clears::AtStart => {
            ecs.app
                .add_observer(clear_after_load)
                .configure_sets(RigSchedule, RigSet::Advance.run_if(startup_cleared));
            ecs.declared_policies = vec!["ClearAtStart".into()];
        }
        Clears::AtSettled => {
            ecs.app.add_observer(clear_after_append);
            ecs.declared_policies = vec!["ClearAtSettled".into()];
        }
    }
    if !ecs.declared_policies.is_empty() {
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert(PolicyVersion(format!(
                "ecs-memory/v1:{}",
                ecs.declared_policies.join("+")
            )));
    }
    let outputs = run_prompts(&mut ecs, prompts, streamed, clears).await;
    for output in &outputs {
        assert!(!output.is_empty());
    }
    ecs.effect_log()
}
