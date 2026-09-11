//! Streaming observations read actual published events and native turn boundaries.
use super::ecs_stress_tools_runtime as tools;
pub(super) use super::ecs_stress_tools_runtime::rewrite_result;
use crate::ecs_agent::EcsAgent;
use bevy_ecs::prelude::*;
use rig::{
    completion::CompletionModel,
    effect::{EffectKind, Outcome},
    streaming::{Delta, StreamEvent},
    tool::ToolResult,
};
use rig_ecs::{
    agent::{Outputs, RequestPatch, Temperature, ToolCallSlot, Turn},
    bus::{BusSet, EffectOutcome, Issued, PendingEffect, RigSchedule, Streamed},
    systems::{Fresh, Materialised, RigSet, spawn_run},
};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

#[derive(Clone, Default)]
pub(super) struct EventTap(Arc<Mutex<Observations>>);
#[derive(Default)]
struct Observations {
    counts: BTreeMap<&'static str, usize>,
    streaming: Option<bool>,
}
impl EventTap {
    pub(super) fn count(&self, tag: &str) -> usize {
        self.0
            .lock()
            .expect("observations")
            .counts
            .get(tag)
            .copied()
            .unwrap_or_default()
    }
    pub(super) fn is_streaming(&self) -> Option<bool> {
        self.0.lock().expect("observations").streaming
    }
}
#[derive(Component)]
struct Taps(Vec<EventTap>);
#[derive(Component)]
struct Published(usize);
#[derive(Component)]
struct TurnObserved;
pub(super) fn agent(
    model: impl CompletionModel + 'static,
    preamble: &str,
    name: &str,
    temperature: Option<f64>,
) -> EcsAgent {
    let mut ecs = tools::agent(model, preamble, name, temperature.unwrap_or_default());
    ecs.app
        .world_mut()
        .entity_mut(ecs.agent)
        .insert(Temperature(temperature));
    ecs.app.add_systems(
        RigSchedule,
        (
            observe_publication
                .after(BusSet::Collect)
                .before(BusSet::Judge),
            observe_turn.after(RigSet::Fold).before(RigSet::Judge),
            observe_tools.after(RigSet::Release).before(BusSet::Gate),
        ),
    );
    ecs
}
fn run_for(world: &World, effect: Entity) -> Entity {
    let turn = world.get::<ChildOf>(effect).expect("effect turn").parent();
    assert!(world.get::<Turn>(turn).is_some());
    world.get::<ChildOf>(turn).expect("turn run").parent()
}
fn emit(world: &World, run: Entity, tag: &'static str) {
    let streamed = world
        .get::<rig_ecs::agent::RunStreaming>(run)
        .expect("actual stream mode")
        .0;
    for tap in &world.get::<Taps>(run).expect("actual run taps").0 {
        let mut observed = tap.0.lock().expect("observations");
        observed.streaming = Some(streamed);
        *observed.counts.entry(tag).or_default() += 1;
    }
}
fn observe_publication(world: &mut World) {
    let events: Vec<_> = world
        .query::<(Entity, &Streamed, Option<&Published>)>()
        .iter(world)
        .map(|(entity, stream, read)| {
            (
                entity,
                stream
                    .events
                    .iter()
                    .skip(read.map_or(0, |r| r.0))
                    .cloned()
                    .collect::<Vec<_>>(),
                stream.events.len(),
            )
        })
        .collect();
    for (effect, events, len) in events {
        for event in events {
            if matches!(
                event,
                StreamEvent::BlockDelta {
                    delta: Delta::Text { .. },
                    ..
                }
            ) {
                emit(world, run_for(world, effect), "TextDelta");
            }
        }
        world.entity_mut(effect).insert(Published(len));
    }
    let completions: Vec<_> = world
        .query_filtered::<(Entity, &EffectOutcome), Added<EffectOutcome>>()
        .iter(world)
        .filter(|(_, outcome)| matches!(outcome.0, Ok(Outcome::Completion(_))))
        .map(|(entity, _)| entity)
        .collect();
    for effect in completions {
        emit(world, run_for(world, effect), "CompletionResponse");
    }
}
fn observe_turn(world: &mut World) {
    let turns: Vec<_> = world
        .query_filtered::<
            (Entity, &ChildOf, &Outputs),
            (With<Turn>, Without<Materialised>, Without<TurnObserved>),
        >()
        .iter(world)
        .filter(|(_, _, outputs)| outputs.done)
        .map(|(entity, parent, _)| (entity, parent.parent()))
        .collect();
    for (turn, run) in turns {
        emit(world, run, "ModelTurnFinished");
        world.entity_mut(turn).insert(TurnObserved);
    }
}
fn observe_tools(world: &mut World) {
    let calls: Vec<_> = world
        .query_filtered::<Entity, (With<ToolCallSlot>, Added<PendingEffect>)>()
        .iter(world)
        .collect();
    for effect in calls {
        emit(world, run_for(world, effect), "ToolCall");
    }
}
pub(super) fn patch(ecs: &mut EcsAgent, patch: RequestPatch) {
    ecs.app.add_systems(
        RigSchedule,
        (move |fresh: Query<Entity, Added<Fresh>>, mut commands: Commands| {
            for turn in &fresh {
                commands.entity(turn).insert(patch.clone());
            }
        })
        .after(RigSet::Select)
        .before(RigSet::Assemble),
    );
}
type Unissued<'w, 's> = Query<
    'w,
    's,
    (Entity, &'static PendingEffect),
    (With<ToolCallSlot>, Without<Issued>, Without<EffectOutcome>),
>;
pub(super) fn skip(ecs: &mut EcsAgent, tool: &'static str, reason: &'static str) {
    ecs.app.add_systems(
        RigSchedule,
        (move |pending: Unissued, mut commands: Commands| {
            for (effect, pending) in &pending {
                if matches!(&pending.kind,EffectKind::ToolCall{name,..} if name == tool) {
                    commands
                        .entity(effect)
                        .insert(EffectOutcome(Ok(Outcome::ToolResult {
                            result: ToolResult::skipped(reason),
                        })));
                }
            }
        })
        .in_set(BusSet::Gate),
    );
}
pub(super) async fn prompt(
    ecs: &mut EcsAgent,
    prompt: &str,
    max_turns: usize,
    streamed: bool,
    taps: Vec<EventTap>,
) -> String {
    let run = spawn_run(
        ecs.app.world_mut(),
        ecs.agent,
        &[],
        prompt,
        streamed,
        Some(max_turns),
    );
    ecs.app.world_mut().entity_mut(run).insert(Taps(taps));
    // Shared native consumer waits for stream EOF and actual settlement, rejects
    // every public stream error (including after Final), and requires RunResult.
    ecs.wait_for_success(run).await
}
