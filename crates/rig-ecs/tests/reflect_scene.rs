//! The `reflect` feature's scene: `ReflectedScene` is the world as
//! reflected data, canonical — entities ordered by content, an `Entity` in
//! a component as its index in the scene — so a world and the world its
//! serde scene (`WorldScene`) loads into export the same JSON.
//!
//! | claim | test |
//! |---|---|
//! | a world and the world loaded from its serde scene reflect to the same scene | `a_world_and_its_loaded_scene_reflect_alike` |
//! | the reflected scene names relationships by index: the run's `RunOf` is the agent's index | `entity_references_are_scene_indexes` |

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing
)]

mod run_support;

use bevy_ecs::{prelude::*, reflect::AppTypeRegistry};
use bevy_reflect::{ReflectDeserialize, ReflectSerialize};
use rig_core::message::AssistantContent;
use rig_ecs::{
    agent::{
        Grant, Order, Settled,
        scene::{load_world, save_world},
    },
    bus::IdCounter,
    reflect::ReflectedScene,
    systems::spawn_run,
};
use run_support::*;

const MODEL: &str = "t/model:default";
const ADD: &str = "t/tool:add#0";

fn ran() -> bevy_app::App {
    let mut app = app();
    rig_ecs::reflect::register(&mut app);
    app.world_mut().resource_mut::<IdCounter>().0 = 1;
    let (model, _) = Scripted::new(
        MODEL,
        vec![
            vec![call("c1", "add", serde_json::json!({"x": 1, "y": 2}))],
            vec![AssistantContent::text("3")],
        ],
    );
    let model = register(&mut app, MODEL, model);
    let add = register(&mut app, ADD, Adder::new(ADD));
    let agent = spawn_agent(app.world_mut(), "t", model);
    app.world_mut()
        .entity_mut(agent)
        .insert(rig_ecs::agent::MaxTurns(2));
    app.world_mut()
        .spawn((Grant(add), Order(0), ChildOf(agent)));
    let run = spawn_run(app.world_mut(), agent, &[], "add one and two", false, None);
    tick_until(&mut app, "the run", |world| {
        world.get::<Settled>(run).is_some()
    });
    app
}

fn json(app: &mut bevy_app::App) -> serde_json::Value {
    let scene = ReflectedScene::from_world(app.world_mut());
    let registry = app.world().resource::<AppTypeRegistry>().clone();
    let registry = registry.read();
    scene.to_json(&registry)
}

#[test]
fn a_world_and_its_loaded_scene_reflect_alike() {
    let mut first = ran();
    let before = json(&mut first);
    let saved = save_world(first.world_mut()).expect("serializes");
    let saved: rig_ecs::agent::scene::WorldScene =
        serde_json::from_str(&serde_json::to_string(&saved).unwrap()).unwrap();
    drop(first);

    let mut second = app();
    rig_ecs::reflect::register(&mut second);
    let (model, _) = Scripted::new(MODEL, Vec::new());
    register(&mut second, MODEL, model);
    register(&mut second, ADD, Adder::new(ADD));
    load_world(&saved, second.world_mut()).expect("the handlers are bound");
    let after = json(&mut second);
    assert_eq!(
        before,
        after,
        "{}",
        serde_json::to_string_pretty(&before).unwrap()
    );
    let entities = before.as_array().unwrap();
    assert!(entities.len() > 8, "{} entities", entities.len());
}

#[test]
fn entity_references_are_scene_indexes() {
    let mut app = ran();
    let scene = json(&mut app);
    let entities = scene.as_array().unwrap();
    let agent = entities
        .iter()
        .position(|entity| entity.get("rig_ecs::agent::Owner").is_some())
        .expect("the agent");
    let run = entities
        .iter()
        .find(|entity| entity.get("rig_ecs::agent::Run").is_some())
        .expect("the run");
    assert_eq!(run["rig_ecs::agent::RunOf"], serde_json::json!(agent));
    let grant = entities
        .iter()
        .find(|entity| entity.get("rig_ecs::agent::Grant").is_some())
        .expect("the grant");
    assert_eq!(
        grant["bevy_ecs::hierarchy::ChildOf"],
        serde_json::json!(agent)
    );
}

#[derive(Clone, Component, bevy_reflect::Reflect, serde::Serialize, serde::Deserialize)]
#[reflect(opaque)]
#[reflect(Component, Serialize, Deserialize)]
#[serde(transparent)]
struct OrderedNumbers(Vec<u64>);

#[test]
fn user_numeric_arrays_preserve_order_and_duplicates() {
    use bevy_reflect::TypePath;
    let mut app = app();
    rig_ecs::reflect::register(&mut app);
    app.register_type::<OrderedNumbers>();
    app.world_mut().spawn(OrderedNumbers(vec![3, 1, 2, 1]));
    let scene = json(&mut app);
    let value = scene
        .as_array()
        .unwrap()
        .iter()
        .find_map(|entity| entity.get(OrderedNumbers::type_path()))
        .expect("registered user component is exported");
    assert_eq!(*value, serde_json::json!([3, 1, 2, 1]));
}
