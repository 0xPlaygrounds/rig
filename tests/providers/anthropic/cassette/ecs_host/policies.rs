//! Native custom-effect producers and acknowledgement boundaries, one active run.
use crate::goldens::{NOTE_KEY, Note, NoteAck};
use bevy_ecs::prelude::*;
use rig::{
    effect::{CustomEffect, EffectFamily, EffectKind, Outcome},
    error::ErrorKind,
};
use rig_ecs::{
    agent::{Run, Settled, Turn},
    bus::{EffectOutcome, Issued, PendingEffect},
    systems::Fresh,
};
#[derive(Component)]
pub(super) struct NotePending {
    at: &'static str,
    accepted: bool,
    unserved: bool,
}
fn note(commands: &mut Commands, run: Entity, at: &'static str, accepted: bool, unserved: bool) {
    commands.spawn((
        NotePending {
            at,
            accepted,
            unserved,
        },
        PendingEffect::new(
            NOTE_KEY,
            EffectKind::Custom {
                kind: Note::KIND.into(),
                payload: serde_json::to_value(Note { at: at.into() }).expect("note serializes"),
            },
        ),
        ChildOf(run),
    ));
}
pub(super) fn at_start(event: On<Add, Run>, mut commands: Commands) {
    note(&mut commands, event.event().entity, "start", true, false);
}
pub(super) fn twice(event: On<Add, Run>, mut commands: Commands) {
    let run = event.event().entity;
    note(&mut commands, run, "first", false, false);
    note(&mut commands, run, "second", false, false);
}
pub(super) fn unserved(event: On<Add, Run>, mut commands: Commands) {
    note(&mut commands, event.event().entity, "unserved", false, true);
}
#[derive(Component)]
pub(super) struct NotedCompletion;
type UnnotedTurns<'w, 's> =
    Query<'w, 's, (Entity, &'static ChildOf), (With<Fresh>, Without<NotedCompletion>)>;
pub(super) fn at_completion_call(fresh: UnnotedTurns, mut commands: Commands) {
    for (turn, parent) in &fresh {
        note(
            &mut commands,
            parent.parent(),
            "completion_call",
            true,
            false,
        );
        commands.entity(turn).insert(NotedCompletion);
    }
}
pub(super) fn at_outcome(
    event: On<Add, EffectOutcome>,
    effects: Query<(&PendingEffect, &ChildOf)>,
    parents: Query<&ChildOf, With<Turn>>,
    mut commands: Commands,
) {
    let Ok((pending, parent)) = effects.get(event.event().entity) else {
        return;
    };
    if pending.kind.family() == EffectFamily::Tool {
        let run = parents
            .get(parent.parent())
            .expect("tool belongs to turn")
            .parent();
        note(&mut commands, run, "outcome", true, false);
    }
}
pub(super) fn at_settled(event: On<Add, Settled>, mut commands: Commands) {
    note(&mut commands, event.event().entity, "settled", true, false);
}
// Gates apply only to this application's one run. Both startup notes must be
// acknowledged before Advance; a completion-call note precedes Assemble; an
// outcome note precedes Materialise. Bus collection itself is never gated.
pub(super) fn ready(notes: Query<(&NotePending, Option<&EffectOutcome>, Has<Issued>)>) -> bool {
    for (note, outcome, issued) in &notes {
        let Some(outcome) = outcome else { return false };
        if note.unserved {
            let error = outcome.0.as_ref().expect_err("no host serves notes");
            assert_eq!(error.kind, ErrorKind::HandlerUnavailable, "{error:?}");
            assert!(!issued, "unavailable note never reaches a handler");
            continue;
        }
        let outcome = outcome.0.as_ref().expect("the host acknowledges");
        let Outcome::Custom { payload } = outcome else {
            panic!("a note acknowledgement, not {outcome:?}")
        };
        let ack: NoteAck =
            serde_json::from_value(payload.clone()).expect("typed note acknowledgement");
        if note.accepted {
            assert!(ack.accepted && ack.at == note.at, "{ack:?}")
        } else {
            assert_eq!(ack.at, note.at, "acknowledged")
        }
    }
    true
}
