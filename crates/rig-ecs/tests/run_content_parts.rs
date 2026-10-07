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
    let text = Text::new("hello");
    let image = Image {
        data: DocumentSourceKind::Base64("Zg==".into()),
        media_type: Some(ImageMediaType::PNG),
        detail: Some(ImageDetail::High),
        native: None,
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
            }),
            UserContent::Video(Video {
                data: DocumentSourceKind::Url("https://invalid.invalid/video".into()),
                media_type: Some(VideoMediaType::MP4),
                additional_params: Some(serde_json::json!({"video_metadata": {"fps": 1}})),
            }),
            UserContent::Document(Document {
                data: DocumentSourceKind::FileId("file-1".into()).into(),
                media_type: Some(DocumentMediaType::PDF),
                additional_params: Some(serde_json::json!({"document_metadata": true})),
            }),
            UserContent::ToolResult(ToolResult {
                is_error: false,
                call: CallId::from_wire("provider-call-1"),
                name: rig_core::message::ToolName::new("read").expect("tool name"),
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

/// Every block's provider item, an opaque item and the turn's origin, stop
/// and provider message survive the graph and a checkpoint, in order.
#[test]
fn assistant_provider_items_opaque_items_and_origin_round_trip() {
    let call = AssistantContent::ToolCall(ToolCall::new(
        CallId::from_wire("provider-call-1"),
        ToolFunction::new(
            rig_core::message::ToolName::new("lookup").expect("tool name"),
            serde_json::json!({"q":"query"}),
        ),
    ))
    .with_native(serde_json::json!({"type": "function_call", "id": "fc_1", "status": "completed"}));
    let reasoning = AssistantContent::Reasoning(Reasoning::new("thinking")).with_native(
        serde_json::json!({"type": "thinking", "thinking": "thinking", "signature": "sig"}),
    );
    let redacted = AssistantContent::Reasoning(Reasoning {
        redacted: true,
        ..Reasoning::default()
    })
    .with_native(serde_json::json!({"type": "redacted_thinking", "data": "redacted-body"}));
    let opaque = AssistantContent::Opaque(Opaque {
        item: serde_json::json!({"type": "web_search_call", "id": "ws_1", "action": {"query": "rig"}}),
        replay: true,
    });
    let mut message = rig_core::message::AssistantMessage::new(vec![
        reasoning,
        opaque,
        AssistantContent::text("answer")
            .with_native(serde_json::json!({"type": "text", "text": "answer", "citations": []})),
        call,
        redacted,
        AssistantContent::Image(Image {
            data: DocumentSourceKind::Raw(vec![1, 2, 3]),
            ..Default::default()
        }),
    ]);
    message.origin = Some(Origin::new("test.api", "test", "model-a"));
    message.stop = Some(StopReason::ToolUse);
    let parts = MessageParts::Assistant(message);
    let (mut world, entity) = world(parts.clone());
    assert_eq!(
        serde_json::to_vec(&read_message(&world, entity).unwrap().to_message()).unwrap(),
        serde_json::to_vec(&parts.to_message()).unwrap(),
    );
    assert_eq!(read_message(&world, entity).unwrap(), parts);
    assert_checkpoint_round_trip(&mut world, &parts);
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
fn leaf_children_and_nested_results_are_rejected_even_when_removed() {
    use bevy_ecs::system::SystemState;
    use std::collections::BTreeMap;

    for nested_result in [false, true] {
        let (mut world, utterance) = world(MessageParts::User {
            content: vec![UserContent::ToolResult(ToolResult {
                is_error: false,
                call: CallId::from_wire("call"),
                name: rig_core::message::ToolName::new("tool").expect("tool name"),
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
            rig_core::message::Origin::new("test.api", "model", ""),
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
