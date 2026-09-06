//! Ordinary application systems implement each original stopping boundary.
use crate::goldens::{
    CANCEL_ADD_DISPATCH, CANCEL_ADD_OUTCOME, CANCEL_ANSWER, STOP_AFTER_TURN, STOP_AT_ANSWER,
    STOP_ON_TEXT_DELTA, STOP_ON_TOOL_CALL_DELTA,
};
use bevy_ecs::prelude::*;
use rig::{
    effect::{EffectKind, Outcome},
    message::AssistantContent,
    streaming::{Delta, StreamEvent},
};
use rig_ecs::{
    agent::{Cancelled, Failed, Failure, Outputs, RunResult, Settled, Turn},
    bus::{EffectOutcome, Issued, PendingEffect, Streamed},
    systems::Materialised,
};

#[derive(Resource, Default)]
pub(super) struct Terminal(pub std::collections::HashMap<Entity, String>);
pub(super) fn failed(event: On<Add, Failed>, failures: Query<&Failed>, mut seen: ResMut<Terminal>) {
    let failure = &failures
        .get(event.event().entity)
        .expect("actual terminal failure")
        .0;
    let reason = match failure {
        Failure::Cancelled(report) => report.message.clone(),
        other => format!("{other:?}"),
    };
    seen.0
        .insert(event.event().entity, format!("error:{reason}"));
}
pub(super) fn settled(
    event: On<Add, Settled>,
    results: Query<&RunResult>,
    mut seen: ResMut<Terminal>,
) {
    seen.0.insert(
        event.event().entity,
        format!(
            "response:{}",
            results
                .get(event.event().entity)
                .expect("actual terminal answer")
                .0
        ),
    );
}

type Unissued<'w, 's> = Query<
    'w,
    's,
    (&'static ChildOf, &'static PendingEffect),
    (Without<Issued>, Without<EffectOutcome>),
>;
pub(super) fn cancel_dispatch(
    effects: Unissued,
    parents: Query<&ChildOf, With<Turn>>,
    mut commands: Commands,
) {
    for (parent, pending) in &effects {
        if matches!(&pending.kind,EffectKind::ToolCall{name,..} if name=="add") {
            let run = parents.get(parent.parent()).expect("tool turn").parent();
            commands
                .entity(run)
                .insert(Cancelled(CANCEL_ADD_DISPATCH.into()));
        }
    }
}
type Landed<'w, 's> = Query<
    'w,
    's,
    (
        &'static ChildOf,
        &'static PendingEffect,
        &'static EffectOutcome,
    ),
    Added<EffectOutcome>,
>;
pub(super) fn cancel_tool_outcome(
    effects: Landed,
    parents: Query<&ChildOf, With<Turn>>,
    mut commands: Commands,
) {
    for (parent, pending, outcome) in &effects {
        if matches!(&pending.kind,EffectKind::ToolCall{name,..} if name=="add")
            && matches!(outcome.0, Ok(Outcome::ToolResult { .. }))
        {
            let run = parents.get(parent.parent()).expect("tool turn").parent();
            commands
                .entity(run)
                .insert(Cancelled(CANCEL_ADD_OUTCOME.into()));
        }
    }
}
pub(super) fn cancel_answer(
    effects: Landed,
    parents: Query<&ChildOf, With<Turn>>,
    mut commands: Commands,
) {
    for (parent, _, outcome) in &effects {
        if let Ok(Outcome::Completion(response)) = &outcome.0
            && !response
                .choice
                .iter()
                .any(|c| matches!(c, AssistantContent::ToolCall(_)))
        {
            let run = parents.get(parent.parent()).expect("model turn").parent();
            commands.entity(run).insert(Cancelled(CANCEL_ANSWER.into()));
        }
    }
}
type Unread<'w, 's> = Query<
    'w,
    's,
    (&'static ChildOf, &'static Outputs),
    (With<Turn>, Without<Materialised>, Changed<Outputs>),
>;
pub(super) fn stop_after_turn(turns: Unread, mut commands: Commands) {
    for (parent, outputs) in &turns {
        if outputs.done {
            commands
                .entity(parent.parent())
                .insert(Cancelled(STOP_AFTER_TURN.into()));
        }
    }
}
pub(super) fn stop_at_answer(turns: Unread, mut commands: Commands) {
    for (parent, outputs) in &turns {
        if outputs.done
            && !outputs
                .content
                .iter()
                .any(|c| matches!(c, AssistantContent::ToolCall(_)))
        {
            commands
                .entity(parent.parent())
                .insert(Cancelled(STOP_AT_ANSWER.into()));
        }
    }
}

// Read only the published stream, then stop the run before dropping its issued
// effect. Native Cancelled alone deliberately leaves issued work to its handler.
fn stop_stream(world: &mut World, reason: &str, predicate: impl Fn(&StreamEvent) -> bool) {
    let mut query = world.query_filtered::<(Entity, &ChildOf, &Streamed), Without<EffectOutcome>>();
    let stops: Vec<_> = query
        .iter(world)
        .filter(|(_, _, stream)| stream.events.iter().any(&predicate))
        .map(|(entity, parent, _)| (entity, parent.parent()))
        .collect();
    for (effect, turn) in stops {
        let run = world
            .get::<ChildOf>(turn)
            .expect("stream turn belongs to run")
            .parent();
        world.entity_mut(run).insert(Cancelled(reason.into()));
        world.flush();
        assert!(
            matches!(&world.get::<Failed>(run).expect("native cancellation observer").0,Failure::Cancelled(report) if report.message==reason)
        );
        world.despawn(effect);
    }
}
pub(super) fn stop_text_delta(world: &mut World) {
    stop_stream(world, STOP_ON_TEXT_DELTA, |event| {
        matches!(
            event,
            StreamEvent::BlockDelta {
                delta: Delta::Text { .. },
                ..
            }
        )
    });
}
pub(super) fn stop_tool_delta(world: &mut World) {
    stop_stream(world, STOP_ON_TOOL_CALL_DELTA, |event| {
        matches!(
            event,
            StreamEvent::BlockDelta {
                delta: Delta::ToolName { .. } | Delta::ToolArguments { .. },
                ..
            }
        )
    });
}
