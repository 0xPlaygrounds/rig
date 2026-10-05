//! Execution permissions are distinct from the request's advertisements.
//! Synthetic models make denied and unadvertised calls deliberately; provider
//! recordings cannot guarantee these adversarial choices on every recapture.
use crate::run_support;

use bevy_ecs::prelude::*;
use rig_core::effect::HandlerKey;
use rig_ecs::{
    agent::{Failed, Failure, Grant, ToolAccess, Turn},
    systems::RunCommands,
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
    app.world_mut().spawn((Grant(tool), ChildOf(agent)));
    app.world_mut().entity_mut(agent).insert(ToolAccess {
        allowed: Some(BTreeSet::new()),
        ..Default::default()
    });
    let run = app.world_mut().spawn_run(agent, &[], "add", false, Some(2));
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
fn permission_and_binding_changes_affect_replay_identity_and_required_row() {
    use rig_cassette::ecs::identity::{required_row, spec_hash};
    let mut app = app();
    let (model, _) = Capturing::new("model", "ok");
    let model = register(&mut app, "model", model);
    let agent = spawn_agent(app.world_mut(), "test", model);
    let run = app
        .world_mut()
        .spawn_run(agent, &[], "hello", false, Some(1));
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
