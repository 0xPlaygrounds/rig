//! Content graph checkpoints: the binary store travels once per payload,
//! every part is remapped to its utterance, and a bad store or a bad graph
//! is refused before the destination is touched.
use std::any::type_name;

use bevy_ecs::prelude::*;
use rig_core::{
    error::ErrorKind,
    message::{DocumentSourceKind, Image, UserContent},
};
use rig_ecs::agent::content::{binary::*, parts::*};
use rig_ecs::agent::{MessageParts, Utterance};
use rig_ecs::checkpoint::{Checkpoint, RestoreMode, load_world, save_world};

/// A world with one utterance of two texts and two images (one base64,
/// one raw) of the same byte.
fn fixture() -> (World, Entity, MessageParts) {
    let mut world = bare_world();
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

/// A bare world with the bus and the agent installed and the crate's types
/// registered: what a checkpoint saves from and loads into.
fn bare_world() -> World {
    let mut world = World::new();
    rig_ecs::bus::BusPlugin::with_policy(rig_core::serve::ServingPolicy::default())
        .install(&mut world);
    rig_ecs::systems::AgentPlugin::install(&mut world);
    rig_ecs::checkpoint::register_types(&mut world);
    world
}

/// The checkpoint through its wire form.
fn round_trip(checkpoint: &Checkpoint) -> Checkpoint {
    Checkpoint::from_json(&checkpoint.to_json().unwrap()).unwrap()
}

/// The checkpoint's entities carrying `C`, by index.
fn entities_with<C: Component>(checkpoint: &Checkpoint) -> Vec<usize> {
    checkpoint
        .entities
        .iter()
        .enumerate()
        .filter(|(_, entity)| entity.contains_key(type_name::<C>()))
        .map(|(index, _)| index)
        .collect()
}

/// Loading `checkpoint` fails as a request error and leaves a fresh
/// destination exactly as it was: no entity, no binary store.
fn refused_without_touching_destination(checkpoint: &Checkpoint, what: &str) {
    let mut destination = bare_world();
    destination.remove_resource::<BinaryAssets>();
    let sentinel = destination.spawn_empty().id();
    let before = destination.entities().len();
    let error = load_world(checkpoint, &mut destination, RestoreMode::Strict, []).expect_err(what);
    assert_eq!(error.kind, ErrorKind::Request, "{what}: {error:?}");
    assert_eq!(destination.entities().len(), before, "{what}");
    assert!(destination.get_entity(sentinel).is_ok(), "{what}");
    assert!(
        !destination.contains_resource::<BinaryAssets>(),
        "{what}: the store was installed"
    );
}

#[test]
fn checkpoints_save_payload_once_and_remap_every_child() {
    let (mut world, _, parts) = fixture();
    let saved = save_world(&mut world).unwrap();
    assert_eq!(saved.binaries.len(), 1, "one payload for two spellings");
    let encoded = saved.to_json().unwrap();
    assert_eq!(
        encoded.matches("Zg==").count(),
        1,
        "the payload occurs once"
    );
    let saved = Checkpoint::from_json(&encoded).unwrap();
    let mut restored = bare_world();
    for _ in 0..17 {
        restored.spawn_empty();
    }
    let loaded = load_world(&saved, &mut restored, RestoreMode::Strict, []).unwrap();
    let utterance = loaded.with::<Utterance>(&restored)[0];
    assert_eq!(read_message(&restored, utterance).unwrap(), parts);
    assert_eq!(restored.resource::<BinaryAssets>().byte_len(), 1);
    let children: Vec<Entity> = loaded.with::<ContentPart>(&restored);
    assert_eq!(children.len(), 4);
    assert!(
        children
            .iter()
            .all(|part| restored.get::<ChildOf>(*part).map(ChildOf::parent) == Some(utterance)),
        "every part is the loaded utterance's"
    );
}

#[test]
fn corrupt_hash_missing_handle_and_bad_parent_leave_destination_untouched() {
    let (mut world, _, _) = fixture();
    let original = round_trip(&save_world(&mut world).unwrap());
    let parts = entities_with::<ContentPart>(&original);
    assert_eq!(parts.len(), 4);
    let mut bad_hash = original.clone();
    bad_hash.binaries.first_mut().unwrap().data = "YQ==".into();
    let mut missing = original.clone();
    missing.binaries.clear();
    let mut bad_parent = original.clone();
    bad_parent.entities[parts[0]].remove(type_name::<ChildOf>());
    for (checkpoint, what) in [
        (bad_hash, "a payload that is not its hash"),
        (missing, "a handle without a payload"),
        (bad_parent, "a part without its utterance"),
    ] {
        refused_without_touching_destination(&checkpoint, what);
    }
}

#[test]
fn old_component_checkpoint_format_is_explicitly_refused() {
    let (mut world, _, _) = fixture();
    let mut checkpoint = save_world(&mut world).unwrap();
    checkpoint.format = 1;
    let error = Checkpoint::from_json(&checkpoint.to_json().unwrap()).unwrap_err();
    assert!(error.message.contains("format 1"));
    assert!(error.message.contains("reads format 2"));
    refused_without_touching_destination(&checkpoint, "the old component format");
}

#[test]
fn load_merges_with_live_assets_and_enforces_destination_limits() {
    let (mut source, _, _) = fixture();
    let saved = round_trip(&save_world(&mut source).unwrap());
    let mut destination = bare_world();
    let mut assets = BinaryAssets::with_limits(BinaryLimits {
        per_asset: 1,
        total: 1,
        count: 1,
    });
    let existing = assets.insert(b"x".to_vec()).unwrap();
    destination.insert_resource(assets);
    let count = destination.entities().len();
    let error = load_world(&saved, &mut destination, RestoreMode::Strict, []).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Request);
    assert_eq!(destination.entities().len(), count);
    assert_eq!(
        destination
            .resource::<BinaryAssets>()
            .get(existing)
            .unwrap(),
        b"x"
    );
    assert_eq!(destination.resource::<BinaryAssets>().byte_len(), 1);
}

/// A run's prompt of two spellings of one image: the world's one payload
/// travels once in the checkpoint's store, and the loaded run's prompt
/// reads back with both spellings, through the wire form.
#[test]
fn a_run_with_two_spellings_of_one_image_round_trips_with_one_payload() {
    use rig_ecs::{
        agent::{Prompt, Run},
        bus::{PendingEffect, RigSchedule},
        systems::RunCommands,
    };
    let (mut world, agent) = crate::run_support::open_model_world();
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
    let run = world.spawn_run(agent, &[], prompt.clone(), false, None);
    world.run_schedule(RigSchedule);
    let utterance = crate::run_support::first_utterance(&mut world, run);
    let expected = read_message(&world, utterance).unwrap();
    assert_eq!(
        expected,
        MessageParts::User {
            content: prompt.0.clone()
        },
        "the graph keeps the prompt's spellings"
    );
    let saved = save_world(&mut world).unwrap();
    assert_eq!(
        entities_with::<PendingEffect>(&saved).len(),
        1,
        "the completion effect retains its request"
    );
    assert_eq!(saved.binaries.len(), 1, "one payload for two spellings");
    assert_eq!(saved.binaries[0].data, "Zg==", "stored canonically");
    let encoded = saved.to_json().unwrap();
    let decoded = Checkpoint::from_json(&encoded).unwrap();
    assert_eq!(
        serde_json::to_value(&decoded).unwrap(),
        serde_json::to_value(&saved).unwrap(),
        "the wire form is lossless"
    );
    let (mut restored, _) = crate::run_support::open_model_world();
    let loaded = load_world(&decoded, &mut restored, RestoreMode::Strict, []).unwrap();
    let run = loaded.with::<Run>(&restored)[0];
    let utterance = crate::run_support::first_utterance(&mut restored, run);
    assert_eq!(read_message(&restored, utterance).unwrap(), expected);
    assert_eq!(restored.resource::<BinaryAssets>().byte_len(), 1);
}
