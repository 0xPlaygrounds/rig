//! Main stress observers: execution commits come from actual committed history.
use super::ecs_stress_runtime as context;
pub(super) use super::ecs_stress_runtime::{EventTap as LifecycleRecorder, ScratchpadReader};
use crate::ecs_agent::EcsAgent;
use bevy_ecs::prelude::*;
use rig::{effect::Outcome, message::UserContent, streaming::StreamEvent};
use rig_ecs::{
    agent::{MessageParts, Run, RunResult, Settled, ToolCallSlot, Turn, Utterance},
    bus::{BusSet, EffectOutcome, Issued, PendingEffect, RigSchedule, Streamed},
    systems::RigSet,
};

#[derive(Component, Clone, Default)]
pub(super) struct RunTrace {
    pub(super) events: Vec<&'static str>,
    pub(super) final_text: Option<String>,
}
#[derive(Component)]
struct Published(usize);
pub(super) struct Observation {
    pub(super) output: String,
}
pub(super) fn agent<
    W: rig_core::wire::Wire<Op = rig_core::operation::Completion>,
    T: rig_core::driver::Transport<W>,
>(
    model: rig_core::driver::Model<W, T>,
    preamble: &str,
    name: &str,
    temperature: Option<f64>,
) -> EcsAgent {
    let mut ecs = context::agent(model, preamble, Some(name), temperature);
    ecs.app
        .add_observer(|added: On<Add, Run>, mut commands: Commands| {
            commands
                .entity(added.event().entity)
                .insert(RunTrace::default());
        });
    ecs.app.add_observer(
        |added: On<Add, Settled>, results: Query<&RunResult>, mut traces: Query<&mut RunTrace>| {
            let run = added.event().entity;
            let mut trace = traces.get_mut(run).expect("actual run trace");
            trace.final_text = Some(results.get(run).expect("actual settled result").0.clone());
            trace.events.push("final_response");
        },
    );
    ecs.app.add_systems(
        RigSchedule,
        (
            published.after(BusSet::Collect).before(BusSet::Judge),
            calls.after(RigSet::Release).before(BusSet::Gate),
            committed.after(RigSet::Materialise).before(RigSet::Settle),
        ),
    );
    ecs
}
fn run_for(world: &World, effect: Entity) -> Entity {
    let turn = world.get::<ChildOf>(effect).expect("effect turn").parent();
    world.get::<ChildOf>(turn).expect("turn run").parent()
}
fn published(world: &mut World) {
    let streams: Vec<_> = world
        .query::<(Entity, &Streamed, Option<&Published>)>()
        .iter(world)
        .map(|(entity, stream, read)| {
            (
                entity,
                stream
                    .events
                    .events()
                    .skip(read.map_or(0, |r| r.0))
                    .cloned()
                    .collect::<Vec<_>>(),
                stream.events.len(),
            )
        })
        .collect();
    for (effect, events, len) in streams {
        let run = run_for(world, effect);
        for event in events {
            let tag = match event {
                StreamEvent::Text { .. } => Some("text"),
                StreamEvent::Arguments { .. } => Some("tool_call_delta"),
                _ => None,
            };
            if let Some(tag) = tag {
                world
                    .get_mut::<RunTrace>(run)
                    .expect("run trace")
                    .events
                    .push(tag);
            }
        }
        world.entity_mut(effect).insert(Published(len));
    }
}
fn calls(world: &mut World) {
    let mut calls: Vec<_> = world
        .query_filtered::<(Entity, &ToolCallSlot), Added<PendingEffect>>()
        .iter(world)
        .map(|(e, s)| (s.index, e))
        .collect();
    calls.sort_by_key(|(index, _)| *index);
    for (_, effect) in calls {
        let run = run_for(world, effect);
        world
            .get_mut::<RunTrace>(run)
            .expect("run trace")
            .events
            .push("tool_call");
    }
}
// land_batch publishes one run-owned user utterance only after the entire batch
// succeeds. Observe that real commit, then correlate each result with its issued
// slot and actual outcome. Merely seeing Issued is not an execution commit.
fn committed(world: &mut World) {
    let messages: Vec<_> = world
        .query_filtered::<(Entity, &ChildOf), (With<Utterance>, Added<Utterance>)>()
        .iter(world)
        .filter_map(|(entity, parent)| {
            let order = crate::ecs_agent::sibling_index(world, entity).expect("a child of the run");
            match rig_ecs::agent::content::parts::read_message(world, entity)
                .expect("valid committed graph")
            {
                MessageParts::User { content } => Some((parent.parent(), order, content.clone())),
                _ => None,
            }
        })
        .collect();
    for (run, order, content) in messages {
        for content in content {
            let UserContent::ToolResult(result) = content else {
                continue;
            };
            // Gemini's generated block IDs can repeat across turns. The native
            // turn immediately preceding this history commit owns the call.
            let turns: Vec<(Entity, usize)> = world
                .query_filtered::<(Entity, &ChildOf), With<Turn>>()
                .iter(world)
                .filter(|(_, parent)| parent.parent() == run)
                .map(|(entity, _)| (entity, crate::ecs_agent::sibling_index(world, entity)))
                .filter_map(|(entity, index)| index.map(|index| (entity, index)))
                .collect();
            let turn = turns
                .into_iter()
                .filter(|(_, turn_order)| *turn_order < order)
                .max_by_key(|(_, turn_order)| *turn_order)
                .map(|(entity, _)| entity)
                .expect("committed tool result belongs to an actual turn");
            let slots: Vec<_> = world
                .query::<(Entity, &ToolCallSlot, Has<Issued>, Option<&EffectOutcome>)>()
                .iter(world)
                .filter(|(entity, slot, _, _)| {
                    slot.id == result.call
                        && world.get::<ChildOf>(*entity).expect("tool turn").parent() == turn
                })
                .map(|(_, _, issued, outcome)| (issued, outcome.cloned()))
                .collect();
            assert_eq!(slots.len(), 1, "committed result has one actual tool slot");
            let (issued, outcome) = &slots[0];
            assert!(
                matches!(outcome, Some(EffectOutcome(Ok(Outcome::ToolResult { .. })))),
                "committed result has a successful bus outcome"
            );
            let mut trace = world.get_mut::<RunTrace>(run).expect("run trace");
            if *issued {
                trace.events.push("tool_execution_committed");
            }
            trace.events.push("tool_result");
        }
    }
}
pub(super) async fn prompt(
    ecs: &mut EcsAgent,
    prompt: &str,
    max_turns: usize,
    streamed: bool,
    taps: Vec<LifecycleRecorder>,
    readers: Vec<ScratchpadReader>,
) -> Observation {
    let output = context::prompt_with_mode(ecs, prompt, max_turns, streamed, taps, readers).await;
    let traces: Vec<_> = ecs
        .app
        .world_mut()
        .query::<&RunTrace>()
        .iter(ecs.app.world())
        .cloned()
        .collect();
    assert_eq!(traces.len(), 1, "one active run in this test App");
    Observation { output }
}
