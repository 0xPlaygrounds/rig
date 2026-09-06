//! Main stress observers: execution commits come from actual committed history.
use super::super::tools_support::ToolEventRecorder;
use super::ecs_stress_runtime as context;
pub(super) use super::ecs_stress_runtime::{EventTap as LifecycleRecorder, ScratchpadReader};
pub(super) use super::ecs_stress_streaming_runtime::{patch, skip};
use crate::ecs_agent::EcsAgent;
use bevy_ecs::prelude::*;
use rig::{
    completion::CompletionModel,
    effect::{EffectKind, Outcome},
    message::UserContent,
    streaming::{Delta, StreamEvent},
    tool::ToolOutput,
};
use rig_ecs::{
    agent::{MessageParts, Order, Parts, Run, RunResult, Settled, ToolCallSlot, Turn, Utterance},
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
    pub(super) trace: RunTrace,
}
pub(super) fn agent(
    model: impl CompletionModel + 'static,
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
                    .iter()
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
                StreamEvent::BlockDelta {
                    delta: Delta::Text { .. },
                    ..
                } => Some("text"),
                StreamEvent::BlockDelta {
                    delta: Delta::ToolName { .. } | Delta::ToolArguments { .. },
                    ..
                } => Some("tool_call_delta"),
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
        .query_filtered::<(&ChildOf, &Order, &Parts), (With<Utterance>, Added<Parts>)>()
        .iter(world)
        .filter_map(|(parent, order, parts)| match &parts.0 {
            MessageParts::User { content } => Some((parent.parent(), order.0, content.clone())),
            _ => None,
        })
        .collect();
    for (run, order, content) in messages {
        for content in content {
            let UserContent::ToolResult(result) = content else {
                continue;
            };
            // Gemini's generated block IDs can repeat across turns. The native
            // turn immediately preceding this history commit owns the call.
            let turn = world
                .query_filtered::<(Entity, &ChildOf, &Order), With<Turn>>()
                .iter(world)
                .filter(|(_, parent, turn_order)| parent.parent() == run && turn_order.0 < order)
                .max_by_key(|(_, _, turn_order)| turn_order.0)
                .map(|(entity, _, _)| entity)
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
type NewTools<'w, 's> = Query<
    'w,
    's,
    &'static mut PendingEffect,
    (
        With<ToolCallSlot>,
        Added<PendingEffect>,
        Without<Issued>,
        Without<EffectOutcome>,
    ),
>;
type Results<'w, 's> = Query<
    'w,
    's,
    (&'static PendingEffect, &'static mut EffectOutcome),
    (With<ToolCallSlot>, Added<EffectOutcome>),
>;
pub(super) fn force_observe_redact(
    ecs: &mut EcsAgent,
    tool: &'static str,
    args: serde_json::Value,
    recorder: ToolEventRecorder,
    marker: &'static str,
) {
    let call_recorder = recorder.clone();
    let force = move |mut tools: NewTools| {
        for mut pending in &mut tools {
            if let EffectKind::ToolCall {
                name,
                args: current,
            } = &mut pending.kind
                && name == tool
            {
                *current = args.to_string();
            }
        }
    };
    let observe = move |tools: NewTools| {
        for pending in &tools {
            if let EffectKind::ToolCall { name, args } = &pending.kind {
                call_recorder
                    .calls
                    .lock()
                    .expect("calls")
                    .push((name.clone(), args.clone()));
            }
        }
    };
    ecs.app
        .add_systems(RigSchedule, (force, observe).chain().in_set(BusSet::Gate));
    let observe = move |tools: Results| {
        for (pending, outcome) in &tools {
            if let EffectKind::ToolCall { name, args } = &pending.kind
                && let Ok(Outcome::ToolResult { result }) = &outcome.0
            {
                recorder.results.lock().expect("results").push((
                    name.clone(),
                    args.clone(),
                    result.output().render(),
                ));
            }
        }
    };
    let redact = move |mut tools: Results| {
        for (pending, mut outcome) in &mut tools {
            if matches!(&pending.kind,EffectKind::ToolCall{name,..} if name==tool)
                && let Ok(Outcome::ToolResult { result }) = &mut outcome.0
            {
                *result = result.clone().with_output(ToolOutput::text(marker));
            }
        }
    };
    ecs.app
        .add_systems(RigSchedule, (observe, redact).chain().in_set(BusSet::Judge));
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
    Observation {
        output,
        trace: traces[0].clone(),
    }
}
