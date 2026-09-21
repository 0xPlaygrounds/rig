//! The checkpoint is canonical: the world as reflected data, entities in
//! `Children` order, an `Entity` in a component as its index in the
//! checkpoint — so a world and the world its checkpoint loads into save the
//! same JSON, whatever entity ids the destination had handed out first.
//!
//! | claim | test |
//! |---|---|
//! | a world and the world loaded from its checkpoint save the same JSON | `a_world_and_its_loaded_checkpoint_save_alike` |
//! | the checkpoint names relationships by index: the run's `RunOf` is the agent's index, every index in range | `entity_references_are_checkpoint_indexes` |
//! | a host's opaque component is exported as its serde form | `user_numeric_arrays_preserve_order_and_duplicates` |

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::indexing_slicing)]

mod run_support;

use std::any::type_name;

use bevy_ecs::prelude::*;
use bevy_reflect::{Reflect, ReflectDeserialize, ReflectSerialize, TypePath};
use rig_core::message::AssistantContent;
use rig_ecs::{
    agent::{Grant, Owner, Run, RunOf, Settled},
    bus::{IdCounter, PendingEffect, Seq},
    checkpoint::{Checkpoint, RestoreMode, load_world, save_world},
    systems::RunCommands,
};
use run_support::*;

const MODEL: &str = "t/model:default";
const ADD: &str = "t/tool:add#0";

/// The handlers every world of this suite serves.
fn serve(app: &mut bevy_app::App, script: Vec<Vec<AssistantContent>>) -> Entity {
    let (model, _) = Scripted::new(MODEL, script);
    let model = register(app, MODEL, model);
    register(app, ADD, Adder::new(ADD));
    model
}

/// A world whose run called a tool and settled.
fn ran() -> bevy_app::App {
    let mut app = app();
    app.world_mut().resource_mut::<IdCounter>().0 = 1;
    let model = serve(
        &mut app,
        vec![
            vec![call("c1", "add", serde_json::json!({"x": 1, "y": 2}))],
            vec![AssistantContent::text("3")],
        ],
    );
    let add = app
        .world()
        .resource::<rig_ecs::bus::HandlerIndex>()
        .entity(&ADD.into())
        .expect("bound");
    let agent = spawn_agent(app.world_mut(), "t", model);
    app.world_mut()
        .entity_mut(agent)
        .insert(rig_ecs::agent::MaxTurns(2));
    app.world_mut().spawn((Grant(add), ChildOf(agent)));
    let run = app
        .world_mut()
        .spawn_run(agent, &[], "add one and two", false, None);
    tick_until(&mut app, "the run", |world| {
        world.get::<Settled>(run).is_some()
    });
    app
}

/// The checkpoint's entities carrying `C`, by index.
fn entities_with<C: Component + TypePath>(checkpoint: &Checkpoint) -> Vec<usize> {
    checkpoint
        .entities
        .iter()
        .enumerate()
        .filter(|(_, entity)| entity.contains_key(C::type_path()))
        .map(|(index, _)| index)
        .collect()
}

/// The checkpoint as a JSON value, and every effect's `Seq` in row order.
fn json_and_seqs(checkpoint: &Checkpoint) -> (serde_json::Value, Vec<u64>) {
    let seqs = checkpoint
        .entities
        .iter()
        .filter_map(|entity| entity.get(type_name::<Seq>()))
        .map(|seq| seq.as_u64().expect("a sequence number"))
        .collect();
    (
        serde_json::from_str(&checkpoint.to_json().unwrap()).unwrap(),
        seqs,
    )
}

#[test]
fn a_world_and_its_loaded_checkpoint_save_alike() {
    let mut first = ran();
    let saved = save_world(first.world_mut()).expect("serializes");
    let saved = Checkpoint::from_json(&saved.to_json().unwrap()).unwrap();
    let (before, seqs) = json_and_seqs(&saved);
    assert!(seqs.len() >= 3, "{} effects", seqs.len());
    assert!(seqs.windows(2).all(|pair| pair[0] < pair[1]), "{seqs:?}");
    drop(first);

    // A fresh world: the same JSON, sequence numbers included.
    let mut second = app();
    serve(&mut second, Vec::new());
    // Entity ids the first world never handed out: an id in the checkpoint
    // would land somewhere else here.
    for _ in 0..7 {
        second.world_mut().spawn_empty();
    }
    let loaded = load_world(&saved, second.world_mut(), RestoreMode::Strict, [])
        .expect("the handlers are bound");
    let again = save_world(second.world_mut()).expect("serializes");
    let (after, _) = json_and_seqs(&again);
    assert_eq!(
        before,
        after,
        "{}",
        serde_json::to_string_pretty(&before).unwrap()
    );
    assert_eq!(
        loaded.entities.len(),
        saved.entities.len(),
        "one entity per checkpoint row"
    );
    assert!(
        saved.entities.len() > 8,
        "{} entities",
        saved.entities.len()
    );

    // A world with effects of its own: the loaded ones come after them, in
    // the saved order with the saved gaps, and the JSON differs by nothing
    // else.
    let mut third = app();
    serve(&mut third, Vec::new());
    let own = third
        .world_mut()
        .spawn(PendingEffect::new(
            ADD,
            rig_core::effect::EffectKind::ToolCall {
                name: "add".into(),
                args: "{\"x\":0,\"y\":0}".into(),
            },
        ))
        .id();
    let own_seq = third.world().get::<Seq>(own).unwrap().0;
    load_world(&saved, third.world_mut(), RestoreMode::Strict, []).expect("the handlers are bound");
    third.world_mut().despawn(own);
    let (mut shifted, seqs_after) = json_and_seqs(&save_world(third.world_mut()).unwrap());
    let offset = seqs_after[0] - seqs[0];
    assert!(
        seqs_after[0] > own_seq,
        "after the world's own: {seqs_after:?}"
    );
    assert_eq!(
        seqs_after
            .iter()
            .map(|seq| seq - offset)
            .collect::<Vec<_>>(),
        seqs,
        "the saved order and gaps"
    );
    for entity in shifted["entities"].as_array_mut().unwrap() {
        if let Some(seq) = entity.get_mut(type_name::<Seq>()) {
            *seq = serde_json::json!(seq.as_u64().unwrap() - offset);
        }
    }
    assert_eq!(before, shifted);
}

#[test]
fn entity_references_are_checkpoint_indexes() {
    let mut app = ran();
    let saved = save_world(app.world_mut()).expect("serializes");
    let agent = entities_with::<Owner>(&saved)[0];
    let run = entities_with::<Run>(&saved)[0];
    assert_eq!(
        saved.entities[run][type_name::<RunOf>()],
        serde_json::json!(agent)
    );
    assert_eq!(
        saved.entities[agent][type_name::<rig_ecs::agent::UsesModel>()],
        serde_json::json!(
            entities_with::<rig_ecs::bus::Bound>(&saved)
                .into_iter()
                .find(
                    |row| saved.entities[*row][rig_ecs::bus::Bound::type_path()]["key"]
                        == serde_json::json!(MODEL)
                )
                .expect("the model's handler")
        ),
        "the agent's model is its handler's index"
    );
    let grant = entities_with::<Grant>(&saved)[0];
    assert_eq!(
        saved.entities[grant][type_name::<ChildOf>()],
        serde_json::json!(agent)
    );
    let effects = entities_with::<PendingEffect>(&saved);
    assert!(!effects.is_empty());
    for (row, entity) in saved.entities.iter().enumerate() {
        for key in [
            type_name::<ChildOf>(),
            type_name::<RunOf>(),
            type_name::<Grant>(),
            type_name::<rig_ecs::agent::UsesModel>(),
            type_name::<rig_ecs::bus::ServedBy>(),
        ] {
            if let Some(index) = entity.get(key) {
                let index = index.as_u64().expect("an index") as usize;
                assert!(
                    index < saved.entities.len(),
                    "entity {row}: {key} = {index}"
                );
                assert_ne!(index, row, "entity {row}: {key} names itself");
            }
        }
    }
}

#[derive(Clone, Component, Reflect, serde::Serialize, serde::Deserialize)]
#[reflect(opaque)]
#[reflect(Component, Serialize, Deserialize)]
#[serde(transparent)]
struct OrderedNumbers(Vec<u64>);

#[test]
fn user_numeric_arrays_preserve_order_and_duplicates() {
    let mut app = app();
    app.register_type::<OrderedNumbers>();
    app.world_mut().spawn(OrderedNumbers(vec![3, 1, 2, 1]));
    let saved = save_world(app.world_mut()).expect("serializes");
    let value = saved
        .entities
        .iter()
        .find_map(|entity| entity.get(type_name::<OrderedNumbers>()))
        .expect("registered user component is exported");
    assert_eq!(*value, serde_json::json!([3, 1, 2, 1]));
}

#[derive(Clone, Component, Reflect, serde::Serialize, serde::Deserialize)]
#[reflect(opaque)]
#[reflect(Component, Serialize, Deserialize)]
#[type_path = "host_checkpoint_contract"]
struct StableComponent(u64);

#[test]
fn custom_reflected_path_survives_checkpoint_roundtrip() {
    assert_ne!(StableComponent::type_path(), type_name::<StableComponent>());
    let mut source = app();
    source.register_type::<StableComponent>();
    source.world_mut().spawn(StableComponent(42));
    let saved = save_world(source.world_mut()).unwrap();
    let row = entities_with::<StableComponent>(&saved)[0];
    assert!(!saved.entities[row].contains_key(type_name::<StableComponent>()));
    let saved = Checkpoint::from_json(&saved.to_json().unwrap()).unwrap();
    let mut destination = app();
    destination.register_type::<StableComponent>();
    let loaded = load_world(&saved, destination.world_mut(), RestoreMode::Strict, []).unwrap();
    assert_eq!(
        destination
            .world()
            .get::<StableComponent>(loaded.entities[row])
            .unwrap()
            .0,
        42
    );
    assert_eq!(
        save_world(destination.world_mut())
            .unwrap()
            .to_json()
            .unwrap(),
        saved.to_json().unwrap()
    );
}
