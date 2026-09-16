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

#[test]
fn all_user_kinds_nested_results_and_metadata_round_trip() {
    let text: Text = serde_json::from_value(serde_json::json!({"text":"hello","additional_params":{"annotations":[{"kind":"citation","url":"https://example.org"}]}})).unwrap();
    let image = Image {
        data: DocumentSourceKind::Base64("Zg==".into()),
        media_type: Some(ImageMediaType::PNG),
        detail: Some(ImageDetail::High),
        additional_params: None,
    };
    let parts = MessageParts::User {
        content: vec![
            UserContent::Text(text.clone()),
            UserContent::Image(image.clone()),
            UserContent::Text(text.clone()),
            UserContent::Image(image.clone()),
            UserContent::Audio(Audio {
                data: DocumentSourceKind::Raw(b"f".to_vec()),
                ..Default::default()
            }),
            UserContent::Video(Video {
                data: DocumentSourceKind::Url("https://invalid.invalid/video".into()),
                ..Default::default()
            }),
            UserContent::Document(Document {
                data: DocumentSourceKind::FileId("file-1".into()),
                ..Default::default()
            }),
            UserContent::ToolResult(ToolResult {
                call: ToolCallId::new("call-1").unwrap(),
                provider: None,
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
    assert_eq!(world.query::<&ImagePart>().iter(&world).count(), 3);
    assert_eq!(world.query::<&ContentPart>().iter(&world).count(), 11);
    assert_eq!(world.query::<&JsonPart>().iter(&world).count(), 1);
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
    let (world, entity) = world(parts.clone());
    assert_eq!(
        serde_json::to_vec(&read_message(&world, entity).unwrap().to_message()).unwrap(),
        serde_json::to_vec(&parts.to_message()).unwrap(),
    );
    assert_eq!(read_message(&world, entity).unwrap(), parts);
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
    world.get_mut::<TextPart>(first).unwrap().0.text = "edited".into();
    assert_eq!(world.get::<TextPart>(second).unwrap().0.text, "second");
    world.entity_mut(entity).insert_children(0, &[second]);
    assert_eq!(
        read_message(&world, entity).unwrap(),
        MessageParts::User {
            content: vec![UserContent::text("second"), UserContent::text("edited")]
        }
    );
}

#[test]
fn missing_conflicting_or_wrong_role_components_are_rejected() {
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
        .insert(JsonPart(serde_json::json!(3)));
    assert_eq!(read_message(&world, entity), Err(ContentError::Shape));
    world.entity_mut(child).remove::<TextPart>();
    assert_eq!(read_message(&world, entity), Err(ContentError::Shape));
    world.entity_mut(entity).remove::<Role>();
    assert_eq!(read_message(&world, entity), Err(ContentError::Missing));
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
    assert_eq!(world.query::<&TextPart>().iter(&world).count(), 1);
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
