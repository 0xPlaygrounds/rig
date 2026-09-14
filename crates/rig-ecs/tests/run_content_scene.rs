//! Content graph scene remapping and asset validation before destination mutation.
#![allow(clippy::unwrap_used, clippy::expect_used)]
use bevy_ecs::prelude::*;
use rig_core::message::{DocumentSourceKind, Image, UserContent};
use rig_ecs::agent::content::{binary::*, parts::*};
use rig_ecs::agent::{
    MessageParts, Utterance,
    scene::{RunScene, SceneKind},
};

fn fixture() -> (World, Entity, MessageParts) {
    let mut world = World::new();
    let utterance = world.spawn(Utterance).id();
    let parts = MessageParts::User {
        content: vec![
            UserContent::text("before"),
            UserContent::Image(Image {
                data: DocumentSourceKind::Base64("Zg==".into()),
                ..Default::default()
            }),
            UserContent::text("between"),
            UserContent::Image(Image {
                data: DocumentSourceKind::Raw(b"f".to_vec()),
                ..Default::default()
            }),
        ],
    };
    write_message(&mut world, utterance, parts.clone()).unwrap();
    (world, utterance, parts)
}

#[test]
fn scenes_save_payload_once_and_remap_every_child() {
    let (mut world, _, parts) = fixture();
    let scene = RunScene::save(&mut world).unwrap();
    assert_eq!(scene.binaries.len(), 1);
    let encoded = serde_json::to_string(&scene).unwrap();
    assert_eq!(encoded.matches("Zg==").count(), 1);
    let scene: RunScene = serde_json::from_str(&encoded).unwrap();
    let mut restored = World::new();
    for _ in 0..17 {
        restored.spawn_empty();
    }
    let entities = scene.load(&mut restored).unwrap();
    let utterance = entities
        .into_iter()
        .find(|entity| restored.get::<Utterance>(*entity).is_some())
        .unwrap();
    assert_eq!(read_message(&restored, utterance).unwrap(), parts);
    assert_eq!(restored.resource::<BinaryAssets>().byte_len(), 1);
    assert_eq!(restored.query::<&ContentPart>().iter(&restored).count(), 4);
}

#[test]
fn corrupt_hash_missing_handle_and_bad_order_leave_destination_untouched() {
    let (mut world, _, _) = fixture();
    let original = RunScene::save(&mut world).unwrap();
    let mut bad_hash = original.clone();
    bad_hash.binaries.first_mut().unwrap().data = "YQ==".into();
    let mut missing = original.clone();
    missing.binaries.clear();
    let mut bad_order = original.clone();
    for part in &mut bad_order.entities {
        if part.kind == SceneKind::ContentPart {
            part.components.insert("order".into(), serde_json::json!(0));
        }
    }
    let mut bad_parent = original.clone();
    bad_parent
        .entities
        .iter_mut()
        .find(|entity| entity.kind == SceneKind::ContentPart)
        .unwrap()
        .parent = None;
    for scene in [bad_hash, missing, bad_order, bad_parent] {
        let mut destination = World::new();
        let sentinel = destination.spawn_empty().id();
        let before = destination.entities().len();
        assert!(scene.load(&mut destination).is_err());
        assert_eq!(destination.entities().len(), before);
        assert!(destination.get_entity(sentinel).is_ok());
        assert!(!destination.contains_resource::<BinaryAssets>());
    }
}

#[test]
fn load_merges_with_live_assets_and_enforces_destination_limits() {
    let (mut source, _, _) = fixture();
    let scene = RunScene::save(&mut source).unwrap();
    let mut destination = World::new();
    let mut assets = BinaryAssets::with_limits(BinaryLimits {
        per_asset: 1,
        total: 1,
        count: 1,
    });
    let existing = assets.insert(b"x".to_vec()).unwrap();
    destination.insert_resource(assets);
    let count = destination.entities().len();
    assert!(scene.load(&mut destination).is_err());
    assert_eq!(destination.entities().len(), count);
    assert_eq!(
        destination
            .resource::<BinaryAssets>()
            .get(existing)
            .unwrap(),
        b"x"
    );
}

#[test]
fn caller_constructed_deep_graph_is_refused_before_destination_mutation() {
    let (mut source, _, _) = fixture();
    let mut scene = RunScene::save(&mut source).unwrap();
    let mut nested = serde_json::Value::Null;
    for _ in 0..65 {
        nested = serde_json::Value::Array(vec![nested]);
    }
    scene
        .entities
        .first_mut()
        .unwrap()
        .components
        .insert("unknown".into(), nested);
    let mut destination = World::new();
    let sentinel = destination.spawn_empty().id();
    let before = destination.entities().len();
    assert!(scene.load(&mut destination).is_err());
    assert_eq!(destination.entities().len(), before);
    assert!(destination.get_entity(sentinel).is_ok());
    assert!(!destination.contains_resource::<BinaryAssets>());
}

#[test]
fn whole_scene_pools_effect_copies_and_escapes_application_json() {
    use rig_core::{
        completion::{ModelRef, ProviderCapabilities},
        effect::FamilyDescriptor,
        serve::ServingPolicy,
    };
    use rig_ecs::{
        agent::scene::{WorldScene, save_world},
        agent::{Owner, Prompt, UsesModel},
        bus::{Bus, Handlers, RigSchedule},
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
    let agent = world.spawn((Owner("owner".into()), UsesModel(model))).id();
    let prompt = Prompt(vec![
        UserContent::Image(Image {
            data: DocumentSourceKind::Base64("Zg==".into()),
            ..Default::default()
        }),
        UserContent::Image(Image {
            data: DocumentSourceKind::Base64("Zh".into()),
            ..Default::default()
        }),
    ]);
    spawn_run(&mut world, agent, &[], prompt, false, None);
    world.run_schedule(RigSchedule);
    let mut scene = save_world(&mut world).unwrap();
    assert_eq!(
        scene.effects.effects.len(),
        1,
        "actual completion effect retains its request"
    );
    let extra = serde_json::json!({"$rig_object":{"$rig_binary":{"Binary":{"id":"not-an-asset"}}},"source":{"type":"base64","value":"Zh"},"literal":"Zg==","invalid_unknown":{"type":"base64","value":"not base64!"}});
    scene.extensions.insert(
        0,
        std::collections::BTreeMap::from([("opaque-test".into(), extra)]),
    );
    let original_effects = serde_json::to_value(&scene.effects).unwrap();
    let original_graph = serde_json::to_value(&scene.graph).unwrap();
    let encoded = serde_json::to_string(&scene).unwrap();
    assert_eq!(
        encoded.matches("Zg==").count(),
        1,
        "the payload occurs only in the asset table"
    );
    assert_eq!(
        encoded.matches("Zh").count(),
        0,
        "noncanonical spelling is represented without another payload copy"
    );
    let decoded = WorldScene::from_json(encoded.as_bytes()).unwrap();
    assert_eq!(
        serde_json::to_value(&decoded.effects).unwrap(),
        original_effects
    );
    assert_eq!(
        serde_json::to_value(&decoded.graph).unwrap(),
        original_graph
    );
    assert_eq!(decoded.extensions, scene.extensions);
    let mut bad: serde_json::Value = serde_json::from_str(&encoded).unwrap();
    *bad.get_mut("graph")
        .and_then(|graph| graph.get_mut("binaries"))
        .expect("serialized scene has a binary table") = serde_json::json!([]);
    assert!(serde_json::from_value::<WorldScene>(bad).is_err());
}

#[test]
fn exact_reserved_object_shapes_round_trip_as_literals() {
    use rig_ecs::agent::scene::WorldScene;
    let values = [
        serde_json::json!({"$rig_binary":"literal"}),
        serde_json::json!({"$rig_object":{"$rig_binary":"nested literal"}}),
        serde_json::json!({"type":"raw","value":[1,2,3]}),
        serde_json::json!({"type":"base64","value":"AQID"}),
    ];
    for value in values {
        let mut scene = WorldScene::default();
        scene.extensions.insert(
            0,
            std::collections::BTreeMap::from([("test".into(), value)]),
        );
        let serialized = serde_json::to_vec(&scene).unwrap();
        let restored = WorldScene::from_json(&serialized).unwrap();
        assert_eq!(restored.extensions, scene.extensions);
    }
}

#[test]
fn shared_binary_fanout_is_bounded_before_dto_expansion() {
    let mut source = World::new();
    let mut assets = BinaryAssets::default();
    let id = assets.insert(vec![1; 1024 * 1024]).unwrap();
    source.insert_resource(assets);
    let utterance = source.spawn((Utterance, rig_ecs::agent::Role::User)).id();
    for order in 0..513 {
        source.spawn((
            ContentPart,
            rig_ecs::agent::Order(order),
            ChildOf(utterance),
            ImagePart {
                source: PartSource::Binary {
                    id,
                    encoding: BinaryEncoding::Raw,
                },
                media_type: None,
                detail: None,
                additional_params: None,
            },
        ));
    }
    let scene = RunScene::save(&mut source).unwrap();
    assert_eq!(scene.binaries.len(), 1);
    let mut destination = World::new();
    let sentinel = destination.spawn_empty().id();
    let before = destination.entities().len();
    let error = scene.load(&mut destination).unwrap_err();
    assert!(error.message.contains("expanded scene byte limit"));
    assert_eq!(destination.entities().len(), before);
    assert!(destination.get_entity(sentinel).is_ok());
    assert!(!destination.contains_resource::<BinaryAssets>());
}

#[test]
fn escaped_object_depth_is_checked_before_serialization_succeeds() {
    use rig_ecs::agent::scene::WorldScene;
    for (depth, accepted) in [(20, true), (40, false)] {
        let mut value = serde_json::Value::Null;
        for _ in 0..depth {
            value = serde_json::json!({"$rig_object":value});
        }
        let mut scene = WorldScene::default();
        scene.extensions.insert(
            0,
            std::collections::BTreeMap::from([("nested".into(), value)]),
        );
        let encoded = serde_json::to_vec(&scene);
        assert_eq!(encoded.is_ok(), accepted);
        if let Ok(encoded) = encoded {
            assert_eq!(
                WorldScene::from_json(&encoded).unwrap().extensions,
                scene.extensions
            );
        }
    }
}
