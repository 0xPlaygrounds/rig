//! Typed graph round-trips and malformed content rejection.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use bevy_ecs::prelude::*;
use rig_core::message::*;
use rig_ecs::agent::content::{binary::*, parts::*};
use rig_ecs::agent::{MessageParts, Order, Utterance};

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
    world.entity_mut(first).insert(Order(5));
    assert_eq!(
        read_message(&world, entity).unwrap(),
        MessageParts::User {
            content: vec![UserContent::text("second"), UserContent::text("edited")]
        }
    );
    world.entity_mut(second).insert(Order(5));
    assert_eq!(
        read_message(&world, entity),
        Err(ContentError::DuplicateOrder)
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
    world.entity_mut(child).remove::<Order>();
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
    use rig_core::completion::{ModelRef, ProviderCapabilities};
    use rig_core::effect::{EffectKind, FamilyDescriptor};
    use rig_core::serve::ServingPolicy;
    use rig_ecs::{
        agent::{MaxTurns, Owner, UsesModel},
        bus::{Bus, Handlers, PendingEffect, RigSchedule},
        systems::{install_agent, spawn_run},
    };
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
    let agent = world
        .spawn((Owner("owner".into()), UsesModel(model), MaxTurns(1)))
        .id();
    let run = spawn_run(&mut world, agent, &[], "hello", false, None);
    world.run_schedule(RigSchedule);
    let utterance = world
        .query_filtered::<(Entity, &ChildOf), With<Utterance>>()
        .iter(&world)
        .find(|(_, parent)| parent.parent() == run)
        .unwrap()
        .0;
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
