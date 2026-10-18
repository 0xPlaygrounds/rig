//! A frozen `rig-session/4` session saved with the in-process subagents
//! plugin on and a subagent mid-turn (`fixtures/subagents-session-4`).
//! Opened with the plugin off, the subagent is parked and every saved value
//! the plugin wrote is kept; opened with it on again, the subagent and its
//! values come back. `make_fixture` (ignored) wrote the fixture.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use rig_basics::subagents::{SubagentsPlugin, TASK};
use rig_core::catalog::{Catalog, Connector};
use rig_core::completion::Message;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_ecs::prelude::*;
use rig_ecs::session_log::SessionStore;
use rig_harness::rig_memory::DirStore;
use rig_harness_test_support::{
    Collected, MemoryStore, app_on, collect, commit_message, connect, start_call,
};
use serde_json::json;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/subagents-session-4")
}

/// A copy of the fixture, for a test to open and write.
fn copy(name: &str) -> PathBuf {
    let to = std::env::temp_dir().join(format!("rig-ecs-{name}-{}", std::process::id()));
    fs::remove_dir_all(&to).ok();
    copy_dir(&fixture(), &to);
    to
}

fn copy_dir(from: &Path, to: &Path) {
    fs::create_dir_all(to).ok();
    for entry in fs::read_dir(from).into_iter().flatten().flatten() {
        let path = entry.path();
        let target = to.join(entry.file_name());
        if path.is_dir() {
            copy_dir(&path, &target);
        } else {
            fs::copy(&path, &target).ok();
        }
    }
}

/// Every log of the session in `dir`, by name, as bytes.
fn logs(dir: &Path) -> BTreeMap<String, Vec<u8>> {
    let mut logs = BTreeMap::new();
    for entry in fs::read_dir(dir.join("logs")).into_iter().flatten().flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        logs.insert(name, fs::read(entry.path()).unwrap_or_default());
    }
    logs
}

/// An app on the session in `dir`, with the subagents plugin when `on`,
/// run for a few frames.
fn open(dir: &Path, on: bool, hold: bool) -> App {
    let (mut app, _wakes) = app_on(&MemoryStore::default());
    let store = DirStore::open(dir);
    assert!(store.is_ok(), "{:?}", store.err());
    if let Ok(store) = store {
        app.insert_resource(SessionStore::new(store));
    }
    app.insert_resource(Models(Connector::new(Catalog::default())))
        .add_plugins((AgentPlugin, SessionLogPlugin));
    collect::<Notice>(&mut app);
    // As rig-harness's `load` does for the plugins it loads.
    app.init_resource::<SavedOwners>();
    if hold {
        app.add_systems(AgentPass, hold_turn_calls.in_set(Stage::Gate));
    }
    if on {
        rig_harness::load::<SubagentsPlugin>(&mut app, "rig-basics", "path plugins/rig-basics");
    }
    app.finish();
    for _ in 0..4 {
        app.update();
    }
    app
}

/// Holds every model call of a turn, as an approval nobody gave.
fn hold_turn_calls(calls: Query<Entity, (ReadyToStart, With<TurnCall>)>, mut commands: Commands) {
    for call in &calls {
        commands.spawn(HoldOf(call));
    }
}

/// The subagents of the session: agents another spawned.
fn children(app: &mut App) -> Vec<Entity> {
    let world = app.world_mut();
    let mut children = world.query_filtered::<Entity, (With<Agent>, With<SpawnedBy>)>();
    children.iter(world).collect()
}

/// Every notice the app wrote.
fn notices(app: &App) -> Vec<String> {
    let notices = app.world().get_resource::<Collected<Notice>>();
    notices
        .map(|notices| notices.0.iter().map(|n| n.text.clone()).collect())
        .unwrap_or_default()
}

#[test]
fn a_session_saved_with_subagents_opens_without_them_and_comes_back_with_them() {
    let off = copy("subagents-off");
    let before = logs(&off);
    assert!(before.len() >= 2, "the fixture holds a parent and a child: {before:?}");
    let mut app = open(&off, false, false);
    let child = children(&mut app);
    assert_eq!(child.len(), 1, "the subagent is restored as an agent");
    let parked = child.iter().all(|&c| app.world().get::<Parked>(c).is_some());
    assert!(parked, "the subagent of the missing plugin is parked");
    let world = app.world_mut();
    let mut turns = world.query::<&TurnOf>();
    let carried: Vec<Entity> = turns.iter(world).map(|t| t.0).collect();
    assert!(
        carried.iter().all(|agent| !child.contains(agent)),
        "the parked subagent's turn is not carried on"
    );
    let told = notices(&app);
    assert!(told.iter().any(|n| n.contains("Left 1 helper agent")), "{told:?}");
    assert!(!told.iter().any(|n| n.contains("next save replaces it")), "{told:?}");
    drop(app);

    // Every record the plugin wrote is still there, byte for byte.
    let after = logs(&off);
    for (name, bytes) in &before {
        let now = after.get(name).cloned().unwrap_or_default();
        assert!(now.starts_with(bytes), "the log {name} kept what it held");
    }

    // With the plugin back, the subagent is its own again.
    let mut app = open(&off, true, true);
    let child = children(&mut app);
    assert_eq!(child.len(), 1);
    let back = child.iter().all(|&c| {
        app.world().get::<Parked>(c).is_none()
            && app.world().get::<Name>(c).map(Name::as_str) == Some("ta")
    });
    assert!(back, "the subagent's saved values came back");
    let world = app.world_mut();
    let mut turns = world.query::<&TurnOf>();
    let carried: Vec<Entity> = turns.iter(world).map(|t| t.0).collect();
    assert!(carried.iter().any(|agent| child.contains(agent)), "its turn goes on");
    drop(app);
    fs::remove_dir_all(&off).ok();
}

/// Writes the fixture: a parent whose `task` started subagent `ta`, its
/// model call held, saved with the plugin on.
#[test]
#[ignore = "writes tests/fixtures/subagents-session-4; run by hand to remake it"]
fn make_fixture() {
    let dir = fixture();
    fs::remove_dir_all(&dir).ok();
    fs::create_dir_all(&dir).ok();
    let model = MockCompletionModel::from_stream_turns(Vec::<Vec<MockStreamEvent>>::new());
    let (mut app, _wakes) = app_on(&MemoryStore::default());
    if let Ok(store) = DirStore::open(&dir) {
        app.insert_resource(SessionStore::new(store));
    }
    app.insert_resource(Models(Connector::new(Catalog::default())))
        .add_plugins((AgentPlugin, SessionLogPlugin))
        .add_systems(AgentPass, hold_turn_calls.in_set(Stage::Gate));
    rig_harness::load::<SubagentsPlugin>(&mut app, "rig-basics", "path plugins/rig-basics");
    if let Some(connection) = connect(&model) {
        let choice = ModelChoice(connection.spec.reference());
        app.world_mut().spawn((Agent, connection, choice));
    }
    app.finish();
    app.update();
    let world = app.world_mut();
    let mut roots = world.query_filtered::<Entity, (With<Agent>, Without<SpawnedBy>)>();
    let parent = roots.iter(world).next().unwrap_or(Entity::PLACEHOLDER);
    assert!(commit_message(world, parent, Message::user("Go.")).is_ok());
    let args = json!({ "description": "ta", "prompt": "Agree on a name.", "peers": true });
    let call = start_call(world, parent, (TASK, "ta"), args);
    for _ in 0..8 {
        if call.is_some_and(|call| app.world().get::<Outcome>(call).is_some()) {
            break;
        }
        app.world_mut().run_schedule(AgentPass);
    }
    for _ in 0..3 {
        app.update();
    }
    assert_eq!(children(&mut app).len(), 1);
}
