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
fn ordered_edits_change_only_one_request_and_preserve_history() {
    let (mut world, _, turn, utterance, target) = fixture();
    let original = read_message(&world, utterance).unwrap();
    // Spawned last, positioned first: sibling order, not spawn order,
    // decides which edit is the later one.
    let later = world
        .spawn((
            RequestPartEdit::Text("patched".into()),
            EditTarget(target),
            ChildOf(turn),
        ))
        .id();
    let early = world
        .spawn((
            RequestPartEdit::Text("earlier".into()),
            EditTarget(target),
            ChildOf(turn),
        ))
        .id();
    world.entity_mut(turn).insert_children(0, &[early]);
    world.run_schedule(RigSchedule);
    let request = world
        .query::<&PendingEffect>()
        .iter(&world)
        .find_map(|effect| match &effect.kind {
            EffectKind::Completion { request, .. } => Some(request),
            _ => None,
        })
        .unwrap();
    assert_eq!(
        request.chat_history,
        vec![Message::User {
            content: vec![UserContent::text("patched"), UserContent::text("sibling")]
        }]
    );
    assert_eq!(read_message(&world, utterance).unwrap(), original);
    assert!(world.get_entity(early).is_err());
    assert!(world.get_entity(later).is_err());
    // A later fold starts from the unchanged persistent graph.
    let mut state: bevy_ecs::system::SystemState<ContentGraph> =
        bevy_ecs::system::SystemState::new(&mut world);
    assert_eq!(
        state.get(&world).unwrap().message(utterance).unwrap(),
        original
    );
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
fn nested_edit_target_and_its_result_parent_are_remapped() {
    use rig_core::message::ToolResultContent;

    let (mut world, _, turn, utterance, _) = fixture();
    let original = MessageParts::User {
        content: vec![UserContent::tool_result(
            "call",
            "tool",
            vec![
                ToolResultContent::text("original"),
                ToolResultContent::text("sibling"),
            ],
        )],
    };
    write_message(&mut world, utterance, original.clone()).unwrap();
    let result = world.get::<Children>(utterance).unwrap()[0];
    let target = world.get::<Children>(result).unwrap()[0];
    world.spawn((
        RequestPartEdit::Text("patched".into()),
        EditTarget(target),
        ChildOf(turn),
    ));
    let checkpoint = save_world(&mut world).unwrap();
    let checkpoint = Checkpoint::from_json(&checkpoint.to_json().unwrap()).unwrap();
    let loaded = load_world(&checkpoint, &mut world, RestoreMode::Strict, []).unwrap();
    let link = loaded.with::<RequestPartEdit>(&world)[0];
    let remapped = world.get::<EditTarget>(link).unwrap().0;
    let remapped_result = world.get::<ChildOf>(remapped).unwrap().parent();
    let remapped_utterance = world.get::<ChildOf>(remapped_result).unwrap().parent();
    assert_ne!(remapped, target);
    assert_ne!(remapped_result, result);
    assert!(loaded.entities.contains(&remapped));
    assert!(loaded.entities.contains(&remapped_result));
    world.run_schedule(RigSchedule);
    let expected = Message::User {
        content: vec![UserContent::tool_result(
            "call",
            "tool",
            vec![
                ToolResultContent::text("patched"),
                ToolResultContent::text("sibling"),
            ],
        )],
    };
    let requests: Vec<_> = world
        .query::<&PendingEffect>()
        .iter(&world)
        .filter_map(|effect| match &effect.kind {
            EffectKind::Completion { request, .. } => Some(request),
            _ => None,
        })
        .collect();
    assert_eq!(requests.len(), 2);
    assert!(
        requests
            .iter()
            .all(|request| request.chat_history == vec![expected.clone()])
    );
    assert_eq!(read_message(&world, utterance).unwrap(), original);
    assert_eq!(read_message(&world, remapped_utterance).unwrap(), original);
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

#[derive(Resource)]
struct DenyPart(Entity);

fn deny_targeted_part(
    target: Res<DenyPart>,
    parts: Query<&ContentPart>,
    effects: Query<
        Entity,
        (
            With<PendingEffect>,
            Without<rig_ecs::bus::Issued>,
            Without<rig_ecs::bus::EffectOutcome>,
        ),
    >,
    mut commands: Commands,
) {
    if parts
        .get(target.0)
        .is_ok_and(|part| matches!(part, ContentPart::Text(text) if text.text == "original"))
    {
        for effect in &effects {
            commands
                .entity(effect)
                .insert(rig_ecs::bus::EffectOutcome(Err(
                    rig_core::error::ErrorReport::new(
                        rig_core::error::ErrorKind::Denied,
                        "target part denied",
                    )
                    .with_retryable(false),
                )));
        }
    }
}

#[test]
fn gate_can_deny_a_target_part_without_mutating_history_or_siblings() {
    let (mut world, run, turn, utterance, target) = fixture();
    let original = read_message(&world, utterance).unwrap();
    world.insert_resource(DenyPart(target));
    world.spawn((
        RequestPartEdit::Text("request only".into()),
        EditTarget(target),
        ChildOf(turn),
    ));
    world.resource_mut::<Schedules>().add_systems(
        RigSchedule,
        deny_targeted_part.in_set(rig_ecs::bus::BusSet::Gate),
    );
    for _ in 0..4 {
        world.run_schedule(RigSchedule);
    }
    assert!(world.get::<Failed>(run).is_some());
    assert_eq!(
        world.query::<&rig_ecs::bus::Issued>().iter(&world).count(),
        0
    );
    assert_eq!(read_message(&world, utterance).unwrap(), original);
}

#[test]
fn missing_target_is_rejected_instead_of_silently_ignoring_an_edit() {
    let (mut world, run, turn, _, _) = fixture();
    world.spawn((RequestPartEdit::Remove, ChildOf(turn)));
    world.run_schedule(RigSchedule);
    assert!(world.get::<Failed>(run).is_some());
    assert_eq!(world.query::<&PendingEffect>().iter(&world).count(), 0);
}

#[test]
fn nested_text_edits_preserve_annotations_and_structured_siblings() {
    use rig_core::message::{Text, ToolCallId, ToolResult, ToolResultContent};
    let (mut world, _, _, utterance, _) = fixture();
    let text: Text = serde_json::from_value(
        serde_json::json!({"text":"original","additional_params":{"annotation":"preserved"}}),
    )
    .unwrap();
    let original = MessageParts::User {
        content: vec![UserContent::ToolResult(ToolResult {
            call: ToolCallId::new("call").unwrap(),
            provider: None,
            name: "tool".into(),
            content: vec![
                ToolResultContent::Text(text.clone()),
                ToolResultContent::Json {
                    value: serde_json::json!({"keep":true}),
                },
            ],
        })],
    };
    write_message(&mut world, utterance, original.clone()).unwrap();
    let result = world
        .get::<Children>(utterance)
        .unwrap()
        .iter()
        .next()
        .unwrap();
    let target = world
        .get::<Children>(result)
        .unwrap()
        .iter()
        .next()
        .unwrap();
    let mut state: bevy_ecs::system::SystemState<ContentGraph> =
        bevy_ecs::system::SystemState::new(&mut world);
    let graph = state.get(&world).unwrap();
    assert_eq!(graph.target_utterance(target).unwrap(), utterance);
    let edits =
        std::collections::BTreeMap::from([(target, RequestPartEdit::Text("changed".into()))]);
    let mut changed = text;
    changed.text = "changed".into();
    let expected = MessageParts::User {
        content: vec![UserContent::ToolResult(ToolResult {
            call: ToolCallId::new("call").unwrap(),
            provider: None,
            name: "tool".into(),
            content: vec![
                ToolResultContent::Text(changed),
                ToolResultContent::Json {
                    value: serde_json::json!({"keep":true}),
                },
            ],
        })],
    };
    assert_eq!(graph.message_with(utterance, &edits).unwrap(), expected);
    let removed = graph
        .message_with(
            utterance,
            &std::collections::BTreeMap::from([(target, RequestPartEdit::Remove)]),
        )
        .unwrap();
    assert!(
        matches!(removed, MessageParts::User { content } if matches!(content.first(), Some(UserContent::ToolResult(result)) if result.content == vec![ToolResultContent::Json { value: serde_json::json!({"keep":true}) }]))
    );
    let removed_parent = graph
        .message_with(
            utterance,
            &std::collections::BTreeMap::from([
                (result, RequestPartEdit::Remove),
                (
                    target,
                    RequestPartEdit::Text("ignored under removed parent".into()),
                ),
            ]),
        )
        .unwrap();
    assert_eq!(removed_parent, MessageParts::User { content: vec![] });
    assert_eq!(graph.message(utterance).unwrap(), original);
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
