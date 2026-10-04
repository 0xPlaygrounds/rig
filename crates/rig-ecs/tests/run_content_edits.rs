//! Part-targeted requests retain history and survive checkpoint entity remapping.
use crate::run_support::{first_utterance, open_model_world};

use bevy_ecs::prelude::*;
use rig_core::{
    effect::EffectKind,
    message::{Message, UserContent},
};
use rig_ecs::{
    agent::{Failed, MessageParts, RequestPatch, Turn, Utterance, content::parts::*},
    bus::{PendingEffect, RigSchedule},
    checkpoint::{Checkpoint, RestoreMode, load_world, save_world},
    systems::{Fresh, RunCommands},
};

fn fixture() -> (World, Entity, Entity, Entity, Entity) {
    let (mut world, agent) = open_model_world();
    let run = world.spawn_run(agent, &[], "original", false, None);
    let utterance = first_utterance(&mut world, run);
    write_message(
        &mut world,
        utterance,
        MessageParts::User {
            content: vec![UserContent::text("original"), UserContent::text("sibling")],
        },
    )
    .unwrap();
    let target = world
        .get::<Children>(utterance)
        .unwrap()
        .iter()
        .next()
        .unwrap();
    let turn = world.spawn((Turn, Fresh, ChildOf(run))).id();
    (world, run, turn, utterance, target)
}

#[test]
fn edit_target_is_remapped_with_the_checkpoint() {
    let (mut world, _, turn, _, target) = fixture();
    world.spawn((RequestPartEdit::Remove, EditTarget(target), ChildOf(turn)));
    let checkpoint = save_world(&mut world).unwrap();
    let checkpoint = Checkpoint::from_json(&checkpoint.to_json().unwrap()).unwrap();
    let loaded = load_world(&checkpoint, &mut world, RestoreMode::Strict, []).unwrap();
    let link = loaded.with::<RequestPartEdit>(&world)[0];
    let remapped = world.get::<EditTarget>(link).unwrap().0;
    assert_ne!(remapped, target);
    assert!(loaded.entities.contains(&remapped));
    assert_eq!(
        world.get::<ContentPart>(remapped),
        world.get::<ContentPart>(target)
    );
    world.run_schedule(RigSchedule);
    let requests: Vec<_> = world
        .query::<&PendingEffect>()
        .iter(&world)
        .filter_map(|effect| match &effect.kind {
            EffectKind::Completion { request, .. } => Some(request),
            _ => None,
        })
        .collect();
    assert_eq!(
        requests.len(),
        2,
        "original and remapped runs both assemble"
    );
    assert!(
        requests
            .iter()
            .all(|request| request.chat_history == vec![Message::user("sibling")])
    );
}

#[test]
fn conflicting_history_and_cross_run_targets_fail_before_dispatch() {
    for cross_run in [false, true] {
        let (mut world, run, turn, _, target) = fixture();
        let target = if cross_run {
            let utterance = world.spawn(Utterance).id();
            write_message(
                &mut world,
                utterance,
                MessageParts::User {
                    content: vec![UserContent::text("foreign")],
                },
            )
            .unwrap();
            world
                .get::<Children>(utterance)
                .unwrap()
                .iter()
                .next()
                .unwrap()
        } else {
            world.entity_mut(turn).insert(RequestPatch {
                history: Some(vec![]),
                ..Default::default()
            });
            target
        };
        world.spawn((RequestPartEdit::Remove, EditTarget(target), ChildOf(turn)));
        world.run_schedule(RigSchedule);
        assert!(world.get::<Failed>(run).is_some());
        assert_eq!(world.query::<&PendingEffect>().iter(&world).count(), 0);
        assert!(matches!(
            world.get::<ContentPart>(target),
            Some(ContentPart::Text(_))
        ));
    }
}

#[test]
fn text_edits_on_images_fail_before_dispatch() {
    let (mut world, run, turn, utterance, _) = fixture();
    write_message(
        &mut world,
        utterance,
        MessageParts::User {
            content: vec![UserContent::Image(rig_core::message::Image::default())],
        },
    )
    .unwrap();
    let target = world
        .get::<Children>(utterance)
        .unwrap()
        .iter()
        .next()
        .unwrap();
    let original = read_message(&world, utterance).unwrap();
    world.spawn((
        RequestPartEdit::Text("invalid".into()),
        EditTarget(target),
        ChildOf(turn),
    ));
    world.run_schedule(RigSchedule);
    assert!(world.get::<Failed>(run).is_some());
    assert_eq!(world.query::<&PendingEffect>().iter(&world).count(), 0);
    assert_eq!(read_message(&world, utterance).unwrap(), original);
}
