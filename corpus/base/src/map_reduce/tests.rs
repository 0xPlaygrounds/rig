use std::collections::HashMap;
use std::path::PathBuf;

use rig_basics::coding_tools::{EditTool, ReadTool, WriteTool};
use rig_core::catalog::{Catalog, Connector};
use rig_core::test_utils::{MockCompletionModel, MockError, MockStreamEvent};
use rig_harness_test_support::MemoryStore;
use serde_json::json;

use super::*;
use rig_harness_test_support::{app_on, calls, connect, reply, run_until, start_call};

/// Each file's scripted model, by file name.
#[derive(Resource, Default)]
struct Scripts(HashMap<String, MockCompletionModel>);

/// Workers alive now, and the most alive at once.
#[derive(Resource, Default)]
struct Alive {
    now: usize,
    most: usize,
}

/// A worker's file name.
fn file_name(worker: &MapWorker) -> String {
    let name = worker.path.as_ref().and_then(|path| path.file_name());
    name.map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Answers each worker by its file's script, and counts the workers alive.
fn on_spawn(
    added: On<Add<MapWorker>>,
    workers: Query<&MapWorker>,
    scripts: Res<Scripts>,
    mut alive: ResMut<Alive>,
    mut commands: Commands,
) {
    alive.now += 1;
    alive.most = alive.most.max(alive.now);
    let model = workers
        .get(added.entity)
        .ok()
        .and_then(|worker| scripts.0.get(&file_name(worker)));
    if let Some(connection) = model.and_then(connect) {
        commands.entity(added.entity).insert(connection);
    }
}

fn on_despawn(_: On<Remove<MapWorker>>, mut alive: ResMut<Alive>) {
    alive.now = alive.now.saturating_sub(1);
}

#[test]
fn five_files_two_at_a_time_one_fails_and_writes_elsewhere_are_refused() {
    let target = std::env::var_os("CARGO_TARGET_DIR")
        .map_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")), PathBuf::from);
    let dir = target.join(format!("corpus-map-reduce-{}", std::process::id()));
    std::fs::create_dir_all(&dir).ok();
    let names = ["a.txt", "b.txt", "c.txt", "d.txt", "e.txt"];
    for name in names {
        std::fs::write(dir.join(name), format!("body of {name}\n")).ok();
    }
    let elsewhere = dir.join("elsewhere.md");
    let path = |name: &str| dir.join(name).display().to_string();
    let edit = |name: &str| {
        let args = json!({ "path": path(name), "edits": [
            { "old_text": "body", "new_text": "// doc\nbody" }
        ]});
        calls("e1", "edit", args)
    };
    let mut scripts = Scripts::default();
    for name in names {
        let mut turns = vec![edit(name), reply("ok")];
        if name == "a.txt" {
            let write = json!({ "path": elsewhere.display().to_string(), "content": "x" });
            turns.insert(0, calls("w1", "write", write));
        }
        if name == "c.txt" {
            let fail = MockStreamEvent::Error(MockError::request("the model broke"));
            turns = vec![vec![fail]];
        }
        let model = MockCompletionModel::from_stream_turns(turns);
        scripts.0.insert(name.to_owned(), model);
    }
    let a = scripts.0.get("a.txt").cloned();

    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(Models(Connector::new(Catalog::default())))
        .insert_resource(scripts)
        .init_resource::<Alive>()
        .add_plugins((AgentPlugin, SessionLogPlugin))
        .add_plugins((ReadTool, EditTool, WriteTool, MapReducePlugin))
        .add_observer(on_spawn)
        .add_observer(on_despawn);
    let parent_model = MockCompletionModel::from_stream_turns([reply("unused")]);
    let mut parent = app.world_mut().spawn(Agent);
    if let Some(connection) = connect(&parent_model) {
        let choice = ModelChoice(connection.spec.reference());
        parent.insert((connection, choice));
    }
    let parent = parent.id();
    app.finish();
    app.update();

    let glob = format!("{}/*.txt", dir.display());
    let args = json!({ "glob": glob, "instruction": "Add a doc comment.", "concurrency": 2 });
    let call = start_call(app.world_mut(), parent, (MAP_FILES, "m1"), args);
    let call = call.unwrap_or(Entity::PLACEHOLDER);
    run_until(&mut app, &wakes, |world| {
        let mut workers = world.iter_entities().filter(|e| e.contains::<MapWorker>());
        world.get::<Outcome>(call).is_some() && workers.next().is_none()
    });

    let output = app.world().get::<Outcome>(call);
    let said = output.map(|output| output.tool_result().output().render());
    let said = said.unwrap_or_default();
    let read = |name: &str| std::fs::read_to_string(dir.join(name)).unwrap_or_default();
    let documented: Vec<&str> = names
        .into_iter()
        .filter(|name| read(name).starts_with("// doc\n"))
        .collect();
    let a_heard = format!("{:?}", a.map(|a| a.requests()));
    let most = app.world().resource::<Alive>().most;
    let workers_left = app
        .world_mut()
        .query::<&MapWorker>()
        .iter(app.world())
        .count();
    std::fs::remove_dir_all(&dir).ok();

    assert_eq!(most, 2, "{said}");
    assert_eq!(workers_left, 0);
    assert_eq!(documented, ["a.txt", "b.txt", "d.txt", "e.txt"], "{said}");
    assert!(said.starts_with("4 of 5 succeeded; failed: "), "{said}");
    assert!(said.contains("c.txt ("), "{said}");
    assert!(said.contains("a.txt: ok"), "{said}");
    assert!(!elsewhere.exists());
    assert!(a_heard.contains("You may change only"), "{a_heard}");
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(MapReducePlugin);
}
