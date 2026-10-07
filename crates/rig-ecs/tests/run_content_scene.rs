//! Content graph checkpoints: the binary store travels once per payload,
//! every part is remapped to its utterance, and a bad store or a bad graph
//! is refused before the destination is touched.
use std::any::type_name;

use bevy_ecs::prelude::*;
use rig_core::message::{DocumentSourceKind, Image, UserContent};
use rig_ecs::agent::content::{binary::*, parts::*};
use rig_ecs::agent::{MessageParts, Utterance};
use rig_ecs::checkpoint::{Checkpoint, CheckpointError, RestoreMode, load_world, save_world};

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
    assert!(
        !matches!(error, CheckpointError::NotInstalled(_)),
        "{what}: {error:?}"
    );
    assert_eq!(destination.entities().len(), before, "{what}");
    assert!(destination.get_entity(sentinel).is_ok(), "{what}");
    assert!(
        !destination.contains_resource::<BinaryAssets>(),
        "{what}: the store was installed"
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
    bad_parent.entities[parts[0]].shift_remove(type_name::<ChildOf>());
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
    assert!(
        matches!(error, CheckpointError::UnsupportedFormat { found: 1 }),
        "{error:?}"
    );
    refused_without_touching_destination(&checkpoint, "the old component format");
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

/// A document's text is saved as the `String` source checkpoints have
/// always spelled it, and loads as the document's text.
#[test]
fn a_saved_string_document_source_loads_as_its_text() {
    let mut world = bare_world();
    let utterance = world.spawn(Utterance).id();
    let parts = MessageParts::User {
        content: vec![UserContent::document_text("the notes", None)],
    };
    write_message(&mut world, utterance, parts.clone()).unwrap();
    let saved = save_world(&mut world).unwrap().to_json().unwrap();
    assert!(saved.contains(r#"{"String":"the notes"}"#), "{saved}");
    let checkpoint = Checkpoint::from_json(&saved).unwrap();
    let mut destination = bare_world();
    load_world(&checkpoint, &mut destination, RestoreMode::Strict, []).unwrap();
    let mut utterances = destination.query_filtered::<Entity, With<Utterance>>();
    let loaded: Vec<Entity> = utterances.iter(&destination).collect();
    assert_eq!(loaded.len(), 1);
    assert_eq!(read_message(&destination, loaded[0]).unwrap(), parts);
}
