//! Typed graph round-trips and malformed content rejection.
use bevy_ecs::prelude::*;
use rig_core::message::*;
use rig_ecs::agent::content::{binary::*, parts::*};
use rig_ecs::agent::{MessageParts, Role, Utterance};

fn world(parts: MessageParts) -> (World, Entity) {
    let mut world = World::new();
    let entity = world.spawn(Utterance).id();
    write_message(&mut world, entity, parts).unwrap();
    (world, entity)
}

fn assert_checkpoint_round_trip(world: &mut World, expected: &MessageParts) {
    use rig_ecs::checkpoint::{Checkpoint, RestoreMode, load_world, register_types, save_world};

    register_types(world);
    let saved = save_world(world).unwrap();
    let saved = Checkpoint::from_json(&saved.to_json().unwrap()).unwrap();
    let mut restored = World::new();
    rig_ecs::bus::BusPlugin::with_policy(rig_core::serve::ServingPolicy::default())
        .install(&mut restored);
    rig_ecs::systems::AgentPlugin::install(&mut restored);
    register_types(&mut restored);
    // Content-only graph: nothing is bound, so the complete requirement set is empty.
    let loaded = load_world(&saved, &mut restored, RestoreMode::Strict, []).unwrap();
    let utterances = loaded.with::<Utterance>(&restored);
    assert_eq!(utterances.len(), 1);
    assert_eq!(&read_message(&restored, utterances[0]).unwrap(), expected);
}

#[test]
fn all_user_kinds_nested_results_and_metadata_round_trip() {
    let text: Text = serde_json::from_value(serde_json::json!({"text":"hello","additional_params":{"annotations":[{"kind":"citation","url":"https://example.org"}]}})).unwrap();
    let image = Image {
        data: DocumentSourceKind::Base64("Zg==".into()),
        media_type: Some(ImageMediaType::PNG),
        detail: Some(ImageDetail::High),
        additional_params: AdditionalParams::from_entries([(
            "image_metadata",
            serde_json::json!(true),
        )]),
    };
    let parts = MessageParts::User {
        content: vec![
            UserContent::Text(text.clone()),
            UserContent::Image(image.clone()),
            UserContent::Text(text.clone()),
            UserContent::Image(image.clone()),
            UserContent::Audio(Audio {
                data: DocumentSourceKind::Raw(b"f".to_vec()),
                media_type: Some(AudioMediaType::MP3),
                additional_params: AdditionalParams::from_entries([(
                    "audio_metadata",
                    serde_json::json!(true),
                )]),
            }),
            UserContent::Video(Video {
                data: DocumentSourceKind::Url("https://invalid.invalid/video".into()),
                media_type: Some(VideoMediaType::MP4),
                additional_params: AdditionalParams::from_entries([(
                    "video_metadata",
                    serde_json::json!(true),
                )]),
            }),
            UserContent::Document(Document {
                data: DocumentSourceKind::FileId("file-1".into()),
                media_type: Some(DocumentMediaType::PDF),
                additional_params: AdditionalParams::from_entries([(
                    "document_metadata",
                    serde_json::json!(true),
                )]),
            }),
            UserContent::ToolResult(ToolResult {
                call: ToolCallId::new("call-1").unwrap(),
                provider: Some(
                    ProviderCallId::new("provider-call-1")
                        .unwrap()
                        .with_item_id("item-1"),
                ),
                name: "read".into(),
                content: vec![
                    ToolResultContent::Text(text),
                    ToolResultContent::Image(image),
                    ToolResultContent::Json {
                        value: serde_json::json!({"error":false,"items":[1,"two"]}),
                    },
                ],
            }),
        ],
    };
    let (mut world, entity) = world(parts.clone());
    assert_eq!(
        serde_json::to_vec(&read_message(&world, entity).unwrap().to_message()).unwrap(),
        serde_json::to_vec(&parts.to_message()).unwrap(),
    );
    assert_eq!(read_message(&world, entity).unwrap(), parts);
    assert_eq!(world.resource::<BinaryAssets>().len(), 1);
    assert_eq!(
        world
            .query::<&ContentPart>()
            .iter(&world)
            .filter(|part| matches!(part, ContentPart::Image(_)))
            .count(),
        3
    );
    assert_eq!(world.query::<&ContentPart>().iter(&world).count(), 11);
    assert_eq!(
        world
            .query::<&ContentPart>()
            .iter(&world)
            .filter(|part| matches!(part, ContentPart::Json(_)))
            .count(),
        1
    );
    assert_checkpoint_round_trip(&mut world, &parts);
}

#[test]
fn assistant_signatures_reasoning_ids_and_images_round_trip() {
    let call = ToolCall {
        id: ToolCallId::new("call-1").unwrap(),
        provider: Some(
            ProviderCallId::new("provider-call-1")
                .unwrap()
                .with_item_id("item-1"),
        ),
        function: ToolFunction::new("lookup".into(), serde_json::json!({"q":"query"})),
        signature: Some("signed-call".into()),
        additional_params: Some(serde_json::json!({"opaque":"preserved"})),
    };
    let parts = MessageParts::Assistant {
        id: Some("message-1".into()),
        content: vec![
            AssistantContent::Reasoning(Reasoning {
                provider: None,
                id: Some("reasoning-1".into()),
                content: vec![
                    ReasoningContent::Text {
                        text: "thinking".into(),
                        signature: Some("sig".into()),
                    },
                    ReasoningContent::Encrypted("secret-body".into()),
                    ReasoningContent::Redacted {
                        data: "redacted-body".into(),
                    },
                    ReasoningContent::Summary("summary".into()),
                ],
            }),
            AssistantContent::Text(Text::new("answer")),
            AssistantContent::ToolCall(call),
            AssistantContent::Image(Image {
                data: DocumentSourceKind::Raw(vec![1, 2, 3]),
                ..Default::default()
            }),
        ],
    };
    let (mut world, entity) = world(parts.clone());
    assert_eq!(
        serde_json::to_vec(&read_message(&world, entity).unwrap().to_message()).unwrap(),
        serde_json::to_vec(&parts.to_message()).unwrap(),
    );
    assert_eq!(read_message(&world, entity).unwrap(), parts);
    assert_checkpoint_round_trip(&mut world, &parts);
}

#[test]
fn part_edits_do_not_change_siblings_and_order_is_semantic() {
    let parts = MessageParts::User {
        content: vec![UserContent::text("first"), UserContent::text("second")],
    };
    let (mut world, entity) = world(parts);
    let children: Vec<_> = world.get::<Children>(entity).unwrap().iter().collect();
    let first = *children.first().unwrap();
    let second = *children.get(1).unwrap();
    let mut part = world.get_mut::<ContentPart>(first).unwrap();
    let ContentPart::Text(text) = &mut *part else {
        panic!("a text part");
    };
    text.text = "edited".into();
    assert!(
        matches!(world.get::<ContentPart>(second), Some(ContentPart::Text(text)) if text.text == "second")
    );
    world.entity_mut(entity).insert_children(0, &[second]);
    assert_eq!(
        read_message(&world, entity).unwrap(),
        MessageParts::User {
            content: vec![UserContent::text("second"), UserContent::text("edited")]
        }
    );
}

#[test]
fn missing_or_wrong_role_components_are_rejected() {
    let (mut world, entity) = world(MessageParts::User {
        content: vec![UserContent::text("text")],
    });
    let child = world
        .get::<Children>(entity)
        .unwrap()
        .iter()
        .next()
        .unwrap();
    world
        .entity_mut(child)
        .insert(ContentPart::Json(serde_json::json!(3)));
    assert_eq!(read_message(&world, entity), Err(ContentError::Shape));
    world.entity_mut(child).remove::<ContentPart>();
    assert_eq!(read_message(&world, entity), Err(ContentError::Shape));
    world.entity_mut(entity).remove::<Role>();
    assert_eq!(read_message(&world, entity), Err(ContentError::Missing));
}

#[test]
fn replacing_a_variant_cannot_leave_a_conflicting_payload() {
    let (mut world, entity) = world(MessageParts::User {
        content: vec![UserContent::text("text")],
    });
    let child = world
        .get::<Children>(entity)
        .unwrap()
        .iter()
        .next()
        .unwrap();
    world
        .entity_mut(child)
        .insert(ContentPart::Image(ImagePart {
            source: PartSource::Url("https://example.org/image".into()),
            media_type: None,
            additional_params: None,
            detail: None,
        }));
    assert_eq!(
        read_message(&world, entity).unwrap(),
        MessageParts::User {
            content: vec![UserContent::Image(Image {
                data: DocumentSourceKind::Url("https://example.org/image".into()),
                ..Default::default()
            })],
        }
    );
    assert_eq!(world.query::<&ContentPart>().iter(&world).count(), 1);
}

#[test]
fn leaf_children_and_nested_results_are_rejected_even_when_removed() {
    use bevy_ecs::system::SystemState;
    use std::collections::BTreeMap;

    for nested_result in [false, true] {
        let (mut world, utterance) = world(MessageParts::User {
            content: vec![UserContent::ToolResult(ToolResult {
                call: ToolCallId::new("call").unwrap(),
                provider: None,
                name: "tool".into(),
                content: vec![ToolResultContent::text("child")],
            })],
        });
        let parent = world
            .get::<Children>(utterance)
            .unwrap()
            .iter()
            .next()
            .unwrap();
        let child = world
            .get::<Children>(parent)
            .unwrap()
            .iter()
            .next()
            .unwrap();
        if nested_result {
            let result = world.get::<ContentPart>(parent).unwrap().clone();
            world.entity_mut(child).insert(result);
        } else {
            world
                .entity_mut(parent)
                .insert(ContentPart::Text(Text::new("invalid parent")));
        }
        assert_eq!(read_message(&world, utterance), Err(ContentError::Shape));
        let mut state: SystemState<ContentGraph> = SystemState::new(&mut world);
        assert_eq!(
            state.get(&world).unwrap().message_with(
                utterance,
                &BTreeMap::from([(parent, RequestPartEdit::Remove)]),
            ),
            Err(ContentError::Shape)
        );
    }
}

#[test]
fn adding_a_payload_exposes_its_parent_and_sibling_membership() {
    #[derive(Resource, Default)]
    struct Seen(usize);

    let mut world = World::new();
    world.init_resource::<Seen>();
    world.add_observer(
        |added: On<Add, ContentPart>,
         relations: Query<&ChildOf>,
         parents: Query<&Children>,
         mut seen: ResMut<Seen>| {
            let parent = relations.get(added.entity).unwrap().parent();
            assert!(
                parents
                    .get(parent)
                    .is_ok_and(|children| children.iter().any(|child| child == added.entity)),
                "the content relationship must precede payload publication"
            );
            seen.0 += 1;
        },
    );
    let utterance = world.spawn(Utterance).id();
    write_message(
        &mut world,
        utterance,
        MessageParts::User {
            content: vec![
                UserContent::text("outer"),
                UserContent::tool_result("call", "tool", vec![ToolResultContent::text("nested")]),
            ],
        },
    )
    .unwrap();
    assert_eq!(world.resource::<Seen>().0, 3);
}

#[test]
fn removal_cannot_hide_a_missing_nested_binary() {
    use bevy_ecs::system::SystemState;
    use std::collections::BTreeMap;

    let (mut world, utterance) = world(MessageParts::User {
        content: vec![UserContent::tool_result(
            "call",
            "tool",
            vec![ToolResultContent::Image(Image {
                data: DocumentSourceKind::Raw(vec![1, 2, 3]),
                ..Default::default()
            })],
        )],
    });
    let result = world.get::<Children>(utterance).unwrap()[0];
    let image = world.get::<Children>(result).unwrap()[0];
    world.insert_resource(BinaryAssets::default());
    let mut state: SystemState<ContentGraph> = SystemState::new(&mut world);
    for target in [result, image] {
        assert_eq!(
            state.get(&world).unwrap().message_with(
                utterance,
                &BTreeMap::from([(target, RequestPartEdit::Remove)]),
            ),
            Err(ContentError::Binary(BinaryError::Missing))
        );
    }
}

#[test]
fn late_preparation_failure_preserves_assistant_id_and_children() {
    let original = MessageParts::Assistant {
        id: Some("retained-message".into()),
        content: vec![AssistantContent::text("retained")],
    };
    let (mut world, utterance) = world(original.clone());
    let children: Vec<_> = world.get::<Children>(utterance).unwrap().iter().collect();
    let rejected = MessageParts::User {
        content: vec![
            UserContent::text("prepared first"),
            UserContent::tool_result(
                "call",
                "tool",
                vec![
                    ToolResultContent::Image(Image {
                        data: DocumentSourceKind::Raw(vec![1, 2, 3]),
                        ..Default::default()
                    }),
                    ToolResultContent::Image(Image {
                        data: DocumentSourceKind::Base64("invalid!".into()),
                        ..Default::default()
                    }),
                ],
            ),
        ],
    };
    assert_eq!(
        write_message(&mut world, utterance, rejected),
        Err(ContentError::Binary(BinaryError::Base64))
    );
    assert_eq!(read_message(&world, utterance).unwrap(), original);
    assert_eq!(
        world
            .get::<Children>(utterance)
            .unwrap()
            .iter()
            .collect::<Vec<_>>(),
        children
    );
}

#[test]
fn failed_persistent_edit_leaves_existing_children_unchanged() {
    let parts = MessageParts::User {
        content: vec![UserContent::text("retained")],
    };
    let (mut world, entity) = world(parts.clone());
    let before: Vec<_> = world.get::<Children>(entity).unwrap().iter().collect();
    let rejected = MessageParts::User {
        content: vec![UserContent::Image(Image {
            data: DocumentSourceKind::Base64("invalid!".into()),
            ..Default::default()
        })],
    };
    assert_eq!(
        write_message(&mut world, entity, rejected),
        Err(ContentError::Binary(BinaryError::Base64))
    );
    assert_eq!(
        serde_json::to_vec(&read_message(&world, entity).unwrap().to_message()).unwrap(),
        serde_json::to_vec(&parts.to_message()).unwrap(),
    );
    assert_eq!(read_message(&world, entity).unwrap(), parts);
    assert_eq!(
        world
            .get::<Children>(entity)
            .unwrap()
            .iter()
            .collect::<Vec<_>>(),
        before
    );
}

#[test]
fn shared_assets_survive_one_owner_despawn_and_host_pins() {
    let parts = MessageParts::User {
        content: vec![UserContent::Image(Image {
            data: DocumentSourceKind::Raw(vec![1, 2, 3]),
            ..Default::default()
        })],
    };
    let (mut world, first) = world(parts.clone());
    let second = world.spawn(Utterance).id();
    write_message(&mut world, second, parts.clone()).unwrap();
    let id = BinaryId::of(&[1, 2, 3]);
    world.despawn(first);
    collect_binary_assets(&mut world, []).unwrap();
    assert_eq!(read_message(&world, second).unwrap(), parts);
    world.despawn(second);
    collect_binary_assets(&mut world, [id]).unwrap();
    assert_eq!(world.resource::<BinaryAssets>().len(), 1);
    collect_binary_assets(&mut world, []).unwrap();
    assert!(world.resource::<BinaryAssets>().is_empty());
}

#[test]
fn system_reader_observes_the_same_graph_without_a_message_cache() {
    use bevy_ecs::system::SystemState;
    let parts = MessageParts::User {
        content: vec![UserContent::text("read through system param")],
    };
    let (mut world, entity) = world(parts.clone());
    let mut state: SystemState<ContentGraph> = SystemState::new(&mut world);
    assert_eq!(state.get(&world).unwrap().message(entity).unwrap(), parts);
}

#[test]
fn new_runtime_stores_parts_as_children_and_folds_the_same_request() {
    use crate::run_support::{first_utterance, open_model_world};
    use rig_core::effect::EffectKind;
    use rig_ecs::{
        agent::MaxTurns,
        bus::{PendingEffect, RigSchedule},
        systems::RunCommands,
    };
    let (mut world, agent) = open_model_world();
    world.entity_mut(agent).insert(MaxTurns(1));
    let run = world.spawn_run(agent, &[], "hello", false, None);
    world.run_schedule(RigSchedule);
    let utterance = first_utterance(&mut world, run);
    assert_eq!(
        read_message(&world, utterance).unwrap(),
        MessageParts::User {
            content: vec![UserContent::text("hello")]
        }
    );
    assert_eq!(
        world
            .query::<&ContentPart>()
            .iter(&world)
            .filter(|part| matches!(part, ContentPart::Text(_)))
            .count(),
        1
    );
    let request = world
        .query::<&PendingEffect>()
        .iter(&world)
        .find_map(|effect| match &effect.kind {
            EffectKind::Completion { request, .. } => Some(request.clone()),
            _ => None,
        })
        .unwrap();
    assert_eq!(
        request.chat_history,
        vec![rig_core::message::Message::user("hello")]
    );
}

/// Two runs read in one `Materialise` pass: the first's model answers an
/// image the graph refuses (an invalid base64 body), the second's a text.
/// The first ends `Failed(Content)`; the second is read in the same pass
/// and settles on its answer — one run's content error is that run's.
#[test]
fn a_content_failure_ends_its_run_and_the_next_run_is_read_in_the_same_pass() {
    use crate::run_support::open_model_world;
    use rig_core::{
        completion::{CompletionResponse, Usage},
        effect::{EffectKind, Outcome},
    };
    use rig_ecs::{
        agent::{Failed, Failure, RunResult, Settled},
        bus::{EffectOutcome, PendingEffect, RigSchedule},
        systems::RunCommands,
    };
    let (mut world, agent) = open_model_world();
    let first = world.spawn_run(agent, &[], "first", false, None);
    let second = world.spawn_run(agent, &[], "second", false, None);
    world.run_schedule(RigSchedule);
    let effect_of = |world: &mut World, run: Entity| -> Entity {
        let turns: Vec<Entity> = world.get::<Children>(run).unwrap().iter().collect();
        world
            .query::<(Entity, &PendingEffect, &ChildOf)>()
            .iter(world)
            .find(|(_, effect, parent)| {
                matches!(effect.kind, EffectKind::Completion { .. })
                    && turns.contains(&parent.parent())
            })
            .map(|(effect, _, _)| effect)
            .expect("a folded completion")
    };
    let answer = |choice: Vec<AssistantContent>| {
        EffectOutcome(Ok(Outcome::Completion(CompletionResponse::new(
            choice,
            Usage::default(),
            "model",
            serde_json::json!({}),
        ))))
    };
    let refused = AssistantContent::Image(Image {
        data: DocumentSourceKind::Base64("invalid!".into()),
        ..Default::default()
    });
    let first_effect = effect_of(&mut world, first);
    let second_effect = effect_of(&mut world, second);
    world
        .entity_mut(first_effect)
        .insert(answer(vec![AssistantContent::text("look"), refused]));
    world
        .entity_mut(second_effect)
        .insert(answer(vec![AssistantContent::text("fine")]));
    world.run_schedule(RigSchedule);
    assert_eq!(
        world.get::<Failed>(first),
        Some(&Failed(Failure::Content(ContentError::Binary(
            BinaryError::Base64
        ))))
    );
    assert!(
        world.get::<Settled>(second).is_some(),
        "read in the same pass"
    );
    assert_eq!(
        world
            .get::<RunResult>(second)
            .map(|result| result.0.as_str()),
        Some("fine")
    );
}
