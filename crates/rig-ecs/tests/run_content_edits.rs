//! Part-targeted requests retain history and survive scene entity remapping.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use bevy_ecs::prelude::*;
use rig_core::{
    completion::{ModelRef, ProviderCapabilities},
    effect::{EffectKind, FamilyDescriptor},
    message::{Message, UserContent},
    serve::ServingPolicy,
};
use rig_ecs::{
    agent::{
        Failed, MessageParts, Order, Owner, RequestPatch, Turn, UsesModel, Utterance,
        content::parts::*, scene::RunScene,
    },
    bus::{Bus, Handlers, PendingEffect, RigSchedule},
    systems::{Fresh, install_agent, spawn_run},
};

fn fixture() -> (World, Entity, Entity, Entity, Entity) {
    let mut world = World::new();
    Bus::with_policy(ServingPolicy::default()).install(&mut world);
    install_agent(&mut world);
    let model = Handlers::with(&mut world, |handlers| {
        handlers.register_open(
            "model",
            FamilyDescriptor::Completion {
                model: ModelRef::new("model"),
                capabilities: ProviderCapabilities::default(),
            },
        )
    })
    .unwrap()
    .unwrap();
    let agent = world.spawn((Owner("owner".into()), UsesModel(model))).id();
    let run = spawn_run(&mut world, agent, &[], "original", false, None);
    let utterance = world
        .query_filtered::<(Entity, &ChildOf), With<Utterance>>()
        .iter(&world)
        .find(|(_, parent)| parent.parent() == run)
        .unwrap()
        .0;
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
    let turn = world.spawn((Turn, Fresh, Order(100), ChildOf(run))).id();
    (world, run, turn, utterance, target)
}

#[test]
fn ordered_edits_change_only_one_request_and_preserve_history() {
    let (mut world, _, turn, utterance, target) = fixture();
    let original = read_message(&world, utterance).unwrap();
    let early = world
        .spawn((
            RequestPartEdit::Text("earlier".into()),
            EditTarget(target),
            Order(0),
            ChildOf(turn),
        ))
        .id();
    let later = world
        .spawn((
            RequestPartEdit::Text("patched".into()),
            EditTarget(target),
            Order(1),
            ChildOf(turn),
        ))
        .id();
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
fn edit_target_is_remapped_with_the_scene() {
    let (mut world, _, turn, _, target) = fixture();
    world.spawn((
        RequestPartEdit::Remove,
        EditTarget(target),
        Order(0),
        ChildOf(turn),
    ));
    let scene = RunScene::save(&mut world).unwrap();
    let encoded = serde_json::to_vec(&scene).unwrap();
    let scene: RunScene = serde_json::from_slice(&encoded).unwrap();
    let loaded = scene.load(&mut world).unwrap();
    let link = loaded
        .iter()
        .copied()
        .find(|entity| world.get::<RequestPartEdit>(*entity).is_some())
        .unwrap();
    let remapped = world.get::<EditTarget>(link).unwrap().0;
    assert_ne!(remapped, target);
    assert!(loaded.contains(&remapped));
    assert_eq!(
        world.get::<TextPart>(remapped),
        world.get::<TextPart>(target)
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
        world.spawn((
            RequestPartEdit::Remove,
            EditTarget(target),
            Order(0),
            ChildOf(turn),
        ));
        world.run_schedule(RigSchedule);
        assert!(world.get::<Failed>(run).is_some());
        assert_eq!(world.query::<&PendingEffect>().iter(&world).count(), 0);
        assert!(world.get::<TextPart>(target).is_some());
    }
}

#[derive(Resource)]
struct DenyPart(Entity);

#[allow(
    clippy::type_complexity,
    reason = "gate fixture names the full pending-effect filter"
)]
fn deny_targeted_part(
    target: Res<DenyPart>,
    parts: Query<&TextPart>,
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
        .is_ok_and(|part| part.0.text == "original")
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
        Order(0),
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
fn missing_order_is_rejected_instead_of_silently_ignoring_an_edit() {
    let (mut world, run, turn, _, target) = fixture();
    world.spawn((RequestPartEdit::Remove, EditTarget(target), ChildOf(turn)));
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
fn duplicate_edit_orders_and_text_edits_on_images_fail_before_dispatch() {
    for wrong_type in [false, true] {
        let (mut world, run, turn, utterance, target) = fixture();
        let target = if wrong_type {
            write_message(
                &mut world,
                utterance,
                MessageParts::User {
                    content: vec![UserContent::Image(rig_core::message::Image::default())],
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
            target
        };
        let original = read_message(&world, utterance).unwrap();
        world.spawn((
            RequestPartEdit::Text("invalid".into()),
            EditTarget(target),
            Order(0),
            ChildOf(turn),
        ));
        if !wrong_type {
            world.spawn((
                RequestPartEdit::Remove,
                EditTarget(target),
                Order(0),
                ChildOf(turn),
            ));
        }
        world.run_schedule(RigSchedule);
        assert!(world.get::<Failed>(run).is_some());
        assert_eq!(world.query::<&PendingEffect>().iter(&world).count(), 0);
        assert_eq!(read_message(&world, utterance).unwrap(), original);
    }
}
