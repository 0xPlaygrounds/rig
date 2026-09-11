//! Execution permissions are distinct from the request's advertisements.
//! Synthetic models make denied and unadvertised calls deliberately; provider
//! recordings cannot guarantee these adversarial choices on every recapture.
#![allow(clippy::expect_used, clippy::indexing_slicing)]
mod run_support;

use bevy_ecs::prelude::*;
use rig_core::{effect::HandlerKey, message::AssistantContent};
use rig_ecs::{
    agent::{Failed, Failure, Grant, Order, Settled, ToolAccess, Turn},
    systems::spawn_run,
};
use run_support::*;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::atomic::Ordering,
};

#[test]
fn empty_permission_set_denies_an_advertised_executable_tool() {
    let mut app = app();
    let (model, requests) = Scripted::new(
        "model",
        vec![vec![call("c", "add", serde_json::json!({"x": 2, "y": 3}))]],
    );
    let model = register(&mut app, "model", model);
    let tool = Adder::new("adder");
    let peak = tool.peak.clone();
    let tool = register(&mut app, "adder", tool);
    let agent = spawn_agent(app.world_mut(), "test", model);
    app.world_mut()
        .spawn((Grant(tool), Order(0), ChildOf(agent)));
    app.world_mut().entity_mut(agent).insert(ToolAccess {
        allowed: Some(BTreeSet::new()),
        ..Default::default()
    });
    let run = spawn_run(app.world_mut(), agent, &[], "add", false, Some(2));
    tick_until(&mut app, "denied call fails", |world| {
        world.get::<Failed>(run).is_some()
    });
    assert!(
        matches!(app.world().get::<Failed>(run), Some(Failed(Failure::UnknownToolCall { name })) if name == "add")
    );
    assert_eq!(peak.load(Ordering::SeqCst), 0);
    let requests = requests.lock().expect("requests");
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].tools.len(), 1);
    assert_eq!(requests[0].tools[0].name, "add");
    let access = app
        .world_mut()
        .query_filtered::<&ToolAccess, With<Turn>>()
        .single(app.world())
        .expect("turn policy");
    assert_eq!(
        access
            .executable
            .as_ref()
            .expect("bindings")
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["add"]
    );
    assert!(access.allowed.as_ref().expect("permissions").is_empty());
}

#[test]
fn explicit_execution_binding_can_serve_an_unadvertised_tool() {
    let mut app = app();
    let (model, requests) = Scripted::new(
        "model",
        vec![
            vec![call("c", "add", serde_json::json!({"x": 2, "y": 3}))],
            vec![AssistantContent::text("5")],
        ],
    );
    let model = register(&mut app, "model", model);
    let tool = Adder::new("hidden-adder");
    let peak = tool.peak.clone();
    register(&mut app, "hidden-adder", tool);
    let agent = spawn_agent(app.world_mut(), "test", model);
    app.world_mut().entity_mut(agent).insert(ToolAccess {
        executable: Some(BTreeMap::from([(
            "add".into(),
            HandlerKey::from("hidden-adder"),
        )])),
        allowed: None,
    });
    let run = spawn_run(app.world_mut(), agent, &[], "add", false, Some(2));
    tick_until(&mut app, "hidden tool completes", |world| {
        world.get::<Settled>(run).is_some() || world.get::<Failed>(run).is_some()
    });
    assert!(
        app.world().get::<Failed>(run).is_none(),
        "{:?}",
        app.world().get::<Failed>(run)
    );
    assert_eq!(peak.load(Ordering::SeqCst), 1);
    let requests = requests.lock().expect("requests");
    assert_eq!(requests.len(), 2);
    assert!(requests.iter().all(|request| request.tools.is_empty()));
}

#[test]
fn permission_and_binding_changes_affect_replay_identity_and_required_row() {
    use rig_ecs::replay::{required_row, spec_hash};
    let mut app = app();
    let (model, _) = Capturing::new("model", "ok");
    let model = register(&mut app, "model", model);
    let agent = spawn_agent(app.world_mut(), "test", model);
    let run = spawn_run(app.world_mut(), agent, &[], "hello", false, Some(1));
    let initial = spec_hash(app.world_mut(), run).expect("policy hash");
    app.world_mut().entity_mut(run).insert(ToolAccess {
        allowed: Some(BTreeSet::new()),
        ..Default::default()
    });
    let denied = spec_hash(app.world_mut(), run).expect("policy hash");
    assert_ne!(initial, denied);
    let hidden = HandlerKey::from("hidden-adder");
    app.world_mut().entity_mut(run).insert(ToolAccess {
        executable: Some(BTreeMap::from([("add".into(), hidden.clone())])),
        allowed: None,
    });
    assert_ne!(
        denied,
        spec_hash(app.world_mut(), run).expect("policy hash")
    );
    let row = required_row(app.world_mut(), run);
    let mut expected = rig_core::effect::EffectRow::new();
    expected.insert(
        HandlerKey::from("model"),
        rig_core::effect::EffectFamily::Completion,
    );
    expected.insert(hidden, rig_core::effect::EffectFamily::Tool);
    assert_eq!(
        row, expected,
        "an unadvertised handler is still a replay dependency"
    );
}

#[test]
fn unadvertised_execution_binding_cannot_impersonate_output_tool() {
    use rig_ecs::agent::{Output, OutputKind, OutputToolConfig};
    let mut app = app();
    let (model, requests) = Capturing::new("model", "unused");
    let model = register(&mut app, "model", model);
    let agent = spawn_agent(app.world_mut(), "test", model);
    app.world_mut().entity_mut(agent).insert((
        Output {
            mode: OutputKind::Tool,
            schema: Some(serde_json::json!({"type": "object"})),
        },
        OutputToolConfig {
            name: Some("submit".into()),
            ..Default::default()
        },
        ToolAccess {
            executable: Some(BTreeMap::from([(
                "submit".into(),
                HandlerKey::from("hidden"),
            )])),
            allowed: None,
        },
    ));
    let run = spawn_run(app.world_mut(), agent, &[], "extract", false, Some(1));
    tick_until(&mut app, "collision rejected", |world| {
        world.get::<Failed>(run).is_some()
    });
    assert!(
        matches!(app.world().get::<Failed>(run), Some(Failed(Failure::OutputToolCollision { name })) if name == "submit")
    );
    assert!(
        requests.lock().expect("requests").is_empty(),
        "no ambiguous provider request"
    );
}

#[test]
fn turn_snapshot_and_old_execution_dependency_survive_fresh_world() {
    use rig_ecs::agent::{Outputs, Run, scene::WorldScene};
    let mut first = app();
    let (model, _) = Capturing::new("model", "unused");
    let model = register(&mut first, "model", model);
    let agent = spawn_agent(first.world_mut(), "test", model);
    let run = spawn_run(first.world_mut(), agent, &[], "hello", true, Some(1));
    let access = ToolAccess {
        executable: Some(BTreeMap::from([(
            "old".into(),
            HandlerKey::from("old-handler"),
        )])),
        allowed: Some(BTreeSet::from(["old".into()])),
    };
    first.world_mut().spawn((
        Turn,
        Order(99),
        access.clone(),
        Outputs {
            stream_validated: 4,
            usage_recorded: true,
            ..Default::default()
        },
        ChildOf(run),
    ));
    first.world_mut().entity_mut(run).insert(ToolAccess {
        allowed: Some(BTreeSet::new()),
        ..Default::default()
    });
    let row = rig_ecs::replay::required_row(first.world_mut(), run);
    let scene = WorldScene::save(first.world_mut()).expect("save graph");
    let scene =
        serde_json::from_slice(&serde_json::to_vec(&scene).expect("encode")).expect("decode");
    drop(first);
    let mut restored = app();
    let (model, _) = Capturing::new("model", "unused");
    register(&mut restored, "model", model);
    WorldScene::load(&scene, restored.world_mut()).expect("restore graph");
    let run = restored
        .world_mut()
        .query_filtered::<Entity, With<Run>>()
        .single(restored.world())
        .expect("run");
    let (actual, outputs) = restored
        .world_mut()
        .query_filtered::<(&ToolAccess, &Outputs), With<Turn>>()
        .single(restored.world())
        .expect("snapshot");
    assert_eq!(actual, &access);
    assert_eq!(outputs.stream_validated, 4);
    assert!(outputs.usage_recorded);
    assert_eq!(
        rig_ecs::replay::required_row(restored.world_mut(), run),
        row
    );
    let mut expected = rig_core::effect::EffectRow::new();
    expected.insert(
        HandlerKey::from("model"),
        rig_core::effect::EffectFamily::Completion,
    );
    expected.insert(
        HandlerKey::from("old-handler"),
        rig_core::effect::EffectFamily::Tool,
    );
    assert_eq!(
        row, expected,
        "new policy must not drop an in-flight snapshot dependency"
    );
}
